import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter, OrderedDict
from sklearn.preprocessing import normalize
import time
import pickle
from utils import fliplr
class CMA(nn.Module):
    '''
    Cross modal Match Aggregation
    '''
    def __init__(self, args):
        super(CMA, self).__init__()
        # self.inited = False
        self.device = torch.device(args.device)
        self.not_saved = True
        # self.threshold = 0.8
        self.num_classes = args.num_classes
        self.T = args.temperature # softmax temperature
        self.sigma = args.sigma # momentum update factor
        # UPR-CRE v0.1 options. Disabled by default, so the baseline behavior is unchanged.
        self.upr_cre = getattr(args, "upr_cre", False)
        self.upr_beta = float(getattr(args, "upr_beta", 0.2))
        self.upr_gamma = float(getattr(args, "upr_gamma", 0.5))
        self.upr_margin_weight = float(getattr(args, "upr_margin_weight", 1.0))
        self.upr_warmup_epoch = int(getattr(args, "upr_warmup_epoch", 1))
        self.current_epoch = None
        self.last_rgb_score_raw = None
        self.last_ir_score_raw = None
        self.last_rgb_score_refined = None
        self.last_ir_score_refined = None
        # UPR-CRE v0.2: confidence filtering / relation curriculum.
        # Disabled by default, so baseline and v0.1 behavior are unchanged unless --upr-filter is set.
        self.upr_filter = getattr(args, "upr_filter", False)
        self.upr_filter_start_epoch = int(getattr(args, "upr_filter_start_epoch", 2))
        self.upr_filter_end_epoch = int(getattr(args, "upr_filter_end_epoch", 10))
        self.upr_filter_start_ratio = float(getattr(args, "upr_filter_start_ratio", 0.75))
        self.upr_filter_end_ratio = float(getattr(args, "upr_filter_end_ratio", 1.0))
        self.upr_filter_min_pairs = int(getattr(args, "upr_filter_min_pairs", 40))
        self.last_upr_filter_stats = {}
        # memory of visible and infrared modal
        self.register_buffer('vis_memory',torch.zeros(self.num_classes,2048))
        self.register_buffer('ir_memory',torch.zeros(self.num_classes,2048))

    @torch.no_grad()
    # def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode):
    def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode, rgb_features=None, ir_features=None, rgb_gt=None, ir_gt=None):
    # vis: vis sample(v2i scores or vis features) ir: ir sample
        self.mode = mode
        self.not_saved = False
        if self.mode != 'scores' and self.mode != 'features':
            raise ValueError('invalid mode!')
        elif self.mode == 'scores': # predict scores
            vis = torch.nn.functional.softmax(self.T*vis,dim=1)
            ir = torch.nn.functional.softmax(self.T*ir,dim=1)
        ###############################
        # save features in memory bank
        if rgb_features is not None and ir_features is not None:
            # Prepare empty memory banks on the device
            self.vis_memory = self.vis_memory.to(self.device)
            self.ir_memory = self.ir_memory.to(self.device)
            
            # Get unique labels and process RGB and IR features
            label_set = torch.unique(rgb_ids)
            
            for label in label_set:
                # Select RGB features for the current label
                rgb_mask = (rgb_ids == label)
                ir_mask = (ir_ids == label)
                # .any() check True in bool tensor
                if rgb_mask.any():
                    rgb_selected = rgb_features[rgb_mask]
                    self.vis_memory[label] = rgb_selected.mean(dim=0)
                
                if ir_mask.any():
                    ir_selected = ir_features[ir_mask]
                    self.ir_memory[label] = ir_selected.mean(dim=0)
        ################################
        vis = vis.detach().cpu().numpy()
        ir = ir.detach().cpu().numpy()
        rgb_ids, ir_ids = rgb_ids.cpu(), ir_ids.cpu()

        # -----------------------------
        # Diagnostic information
        # -----------------------------
        self.last_rgb_score = vis
        self.last_ir_score = ir
        
        self.last_rgb_score_raw = vis.copy()
        self.last_ir_score_raw = ir.copy()
        self.last_rgb_score_refined = None
        self.last_ir_score_refined = None

        self.rgb_gt = rgb_gt.detach().cpu() if rgb_gt is not None else None
        self.ir_gt = ir_gt.detach().cpu() if ir_gt is not None else None

        # -----------------------------
        # Existing fields
        # -----------------------------
        self.vis, self.ir = vis, ir
        self.rgb_ids, self.ir_ids = rgb_ids, ir_ids
        self.rgb_idx, self.ir_idx = rgb_idx, ir_idx
        
    @torch.no_grad()
    def update(self, rgb_feats, ir_feats, rgb_labels, ir_labels):
        rgb_set = torch.unique(rgb_labels)
        ir_set = torch.unique(ir_labels)
        for i in rgb_set:
            rgb_mask = (rgb_labels == i)
            selected_rgb = rgb_feats[rgb_mask].mean(dim=0)
            self.vis_memory[i] = (1-self.sigma)*self.vis_memory[i] + self.sigma * selected_rgb
        for i in ir_set:
            ir_mask = (ir_labels == i)
            selected_ir = ir_feats[ir_mask].mean(dim=0)
            self.ir_memory[i] = (1-self.sigma)*self.ir_memory[i] + self.sigma * selected_ir

    def get_label(self, epoch=None):
        self.current_epoch = epoch
        if self.not_saved:# pass if 
            pass
        else:
            print('get match labels')
            if self.mode == 'features':
                dists = np.matmul(self.vis, self.ir.T)
                v2i_dict, i2v_dict = self._get_label(dists,'dist')

            elif self.mode == 'scores':
                v2i_dict, _ = self._get_label(self.vis,'rgb')
                i2v_dict, _ = self._get_label(self.ir,'ir')

                self.v2i = v2i_dict
                self.i2v = i2v_dict

                # diagnostic snapshots
                self.last_v2i = v2i_dict
                self.last_i2v = i2v_dict
            return v2i_dict, i2v_dict
        
    def _upr_enabled_for_epoch(self):
        if not getattr(self, "upr_cre", False):
            return False
        if self.current_epoch is None:
            return True
        return int(self.current_epoch) >= int(self.upr_warmup_epoch)

    def _row_confidence(self, score):
        """Return row-level confidence from entropy and top-1/top-2 margin."""
        eps = 1e-12
        prob = np.asarray(score, dtype=np.float32)
        prob = np.clip(prob, eps, 1.0)

        entropy = -(prob * np.log(prob)).sum(axis=1)
        entropy = entropy / np.log(prob.shape[1] + eps)
        entropy = np.clip(entropy, 0.0, 1.0)

        if prob.shape[1] >= 2:
            top2 = np.partition(prob, -2, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
        else:
            margin = np.ones((prob.shape[0],), dtype=np.float32)
        margin = np.clip(margin, 0.0, 1.0)

        conf = (1.0 - entropy) + float(self.upr_margin_weight) * margin
        return np.clip(conf, 0.0, 1.0).astype(np.float32)

    def _prototype_score_for_rows(self, mode):
        """Return prototype similarity matrix aligned with the score matrix rows."""
        vm = self.vis_memory.detach().float().to(self.device)
        im = self.ir_memory.detach().float().to(self.device)

        # If memory is still empty, avoid injecting uninformative prototype scores.
        if torch.sum(torch.abs(vm)).item() == 0 or torch.sum(torch.abs(im)).item() == 0:
            return None

        vm = torch.nn.functional.normalize(vm, dim=1)
        im = torch.nn.functional.normalize(im, dim=1)
        proto = torch.matmul(vm, im.t())
        proto = ((proto + 1.0) * 0.5).clamp(0.0, 1.0)
        proto = proto.detach().cpu().numpy().astype(np.float32)

        if mode == "rgb":
            # Row is a visible/RGB sample, column is an infrared identity class.
            row_labels = self.rgb_ids.numpy().astype(np.int64)
            return proto[row_labels, :]
        if mode == "ir":
            # Row is an infrared sample, column is a visible/RGB identity class.
            row_labels = self.ir_ids.numpy().astype(np.int64)
            return proto[:, row_labels].T
        return None

    def _apply_upr_cre_score(self, dists, mode):
        """Refine expert score matrix before hard CRE pair selection.

        This v0.1 method does not change the output format of CRE.
        It only changes the matrix used by `_get_label` for ranking.
        """
        if mode not in ("rgb", "ir"):
            return dists
        if not self._upr_enabled_for_epoch():
            return dists

        score = np.asarray(dists, dtype=np.float32)
        proto_score = self._prototype_score_for_rows(mode)
        if proto_score is None or proto_score.shape != score.shape:
            return dists

        beta = float(np.clip(self.upr_beta, 0.0, 1.0))
        gamma = float(np.clip(self.upr_gamma, 0.0, 1.0))

        row_conf = self._row_confidence(score)
        confidence_scale = (1.0 - gamma) + gamma * row_conf[:, None]
        refined = (1.0 - beta) * score * confidence_scale + beta * proto_score
        refined = refined.astype(np.float32)

        if mode == "rgb":
            self.last_rgb_score_refined = refined
            self.last_rgb_score = refined
        elif mode == "ir":
            self.last_ir_score_refined = refined
            self.last_ir_score = refined

        return refined

    def _upr_filter_keep_ratio(self):
        """Curriculum keep ratio for v0.2 relation filtering."""
        if not getattr(self, "upr_filter", False):
            return 1.0

        if self.current_epoch is None:
            return 1.0

        epoch = int(self.current_epoch)
        start_epoch = int(self.upr_filter_start_epoch)
        end_epoch = int(self.upr_filter_end_epoch)
        start_ratio = float(np.clip(self.upr_filter_start_ratio, 0.0, 1.0))
        end_ratio = float(np.clip(self.upr_filter_end_ratio, 0.0, 1.0))

        if epoch < start_epoch:
            return 1.0
        if end_epoch <= start_epoch:
            return end_ratio
        if epoch >= end_epoch:
            return end_ratio

        progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
        return float(start_ratio + progress * (end_ratio - start_ratio))

    def _pair_confidence_scores(self, score_matrix, mode, mapping):
        """Compute a confidence score for each source->target pair in a directional mapping.

        The score is the mean row score over all samples belonging to the source identity.
        This keeps the filter independent of ground-truth correspondence labels.
        """
        if mapping is None or len(mapping) == 0:
            return {}

        score = np.asarray(score_matrix, dtype=np.float32)
        if mode == "rgb":
            src_labels = self.rgb_ids.numpy().astype(np.int64)
        elif mode == "ir":
            src_labels = self.ir_ids.numpy().astype(np.int64)
        else:
            return {src: 1.0 for src in mapping.keys()}

        out = {}
        for src, tgt in mapping.items():
            try:
                src_i = int(src)
                tgt_i = int(tgt)
            except Exception:
                out[src] = 0.0
                continue

            rows = np.where(src_labels == src_i)[0]
            if rows.size == 0 or tgt_i < 0 or tgt_i >= score.shape[1]:
                out[src] = 0.0
            else:
                out[src] = float(np.mean(score[rows, tgt_i]))
        return out

    def _apply_upr_pair_filter(self, score_matrix, mode, v2i, i2v):
        """Filter low-confidence directional pseudo-relations after hard CRE mapping.

        This is UPR-CRE v0.2. It does not change the loss function or relation type.
        It only removes low-confidence source->target pairs before train.py constructs
        common/specific/remain dictionaries.
        """
        if not getattr(self, "upr_filter", False):
            return v2i, i2v

        # Before start epoch, keep original mapping unchanged.
        if self.current_epoch is not None and int(self.current_epoch) < int(self.upr_filter_start_epoch):
            return v2i, i2v

        if v2i is None or len(v2i) == 0:
            return v2i, i2v

        keep_ratio = self._upr_filter_keep_ratio()
        original_n = len(v2i)
        min_pairs = max(0, int(self.upr_filter_min_pairs))
        keep_n = int(np.ceil(original_n * keep_ratio))
        keep_n = max(min_pairs, keep_n)
        keep_n = min(original_n, keep_n)

        pair_conf = self._pair_confidence_scores(score_matrix, mode, v2i)
        ranked_sources = sorted(pair_conf.keys(), key=lambda k: pair_conf[k], reverse=True)
        keep_sources = set(ranked_sources[:keep_n])

        filtered_v2i = OrderedDict()
        for src, tgt in v2i.items():
            if src in keep_sources:
                filtered_v2i[src] = tgt

        filtered_i2v = OrderedDict((tgt, src) for src, tgt in filtered_v2i.items())

        self.last_upr_filter_stats[mode] = {
            "epoch": None if self.current_epoch is None else int(self.current_epoch),
            "mode": mode,
            "enabled": True,
            "keep_ratio": float(keep_ratio),
            "original_pairs": int(original_n),
            "kept_pairs": int(len(filtered_v2i)),
            "min_pairs": int(min_pairs),
            "mean_pair_confidence_before": float(np.mean(list(pair_conf.values()))) if pair_conf else 0.0,
            "mean_pair_confidence_after": float(np.mean([pair_conf[k] for k in keep_sources])) if keep_sources else 0.0,
        }

        return filtered_v2i, filtered_i2v

    def _get_label(self,dists,mode):
        sample_rate = 1
        dists = self._apply_upr_cre_score(dists, mode)
        score_matrix = np.asarray(dists, dtype=np.float32).copy()
        dists_shape = dists.shape
        sorted_1d = np.argsort(dists, axis=None)[::-1]# flat to 1d and sort
        sorted_2d = np.unravel_index(sorted_1d, dists_shape)# sort index return to 2d, like ([0,1,2],[1,2,0])
        idx1, idx2 = sorted_2d[0], sorted_2d[1]# sorted idx of dim0 and dim1
        dists = dists[idx1, idx2]
        idx_length = int(np.ceil(sample_rate*dists.shape[0]/self.num_classes))
        dists = dists[:idx_length]

        if mode=='dist': # multiply the instance features of the two modalities
            convert_label = [(i,j) for i,j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],\
                                            np.array(self.ir_ids)[idx2[:idx_length]])]
            
        elif mode=='rgb': # classify score of RGB (v2i)
            convert_label = [(i,j) for i,j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],\
                                                  idx2[:idx_length])]

        elif mode=='ir': # classify score of IR (v2i)
            convert_label = [(i,j) for i,j in zip(np.array(self.ir_ids)[idx1[:idx_length]],\
                                                  idx2[:idx_length])]
        else:
            raise AttributeError('invalid mode!')
        convert_label_cnt = Counter(convert_label)
        convert_label_cnt_sorted = sorted(convert_label_cnt.items(),key = lambda x:x[1],reverse = True)
        length = len(convert_label_cnt_sorted)
        lambda_cm=0.1
        in_rgb_label=[]
        in_ir_label=[]
        v2i = OrderedDict()
        i2v = OrderedDict()

        length_ratio = 1
        for i in range(int(length*length_ratio)):
            key = convert_label_cnt_sorted[i][0] 
            value = convert_label_cnt_sorted[i][1]
            # if key[0] == -1 or key[1] == -1:
            #     continue
            if key[0] in in_rgb_label or key[1] in in_ir_label:
                continue
            in_rgb_label.append(key[0])
            in_ir_label.append(key[1])
            v2i[key[0]] = key[1]
            i2v[key[1]] = key[0]
            # v2i[key[0]][key[1]] = 1

        v2i, i2v = self._apply_upr_pair_filter(score_matrix, mode, v2i, i2v)
        return v2i, i2v # only v2i/i2v is used in scores mode

    def extract(self, args, model, dataset):
        '''
        Output: BN_features, labels, cls
        '''
        # save epoch
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader() 
        with torch.no_grad():
            
            rgb_features, rgb_labels, rgb_gt, r2i_cls, rgb_idx = self._extract_feature(model, rgb_loader,'rgb')
            ir_features, ir_labels, ir_gt, i2r_cls, ir_idx = self._extract_feature(model, ir_loader,'ir')

        # # //match by cls and save features to memory bank
        self.save(
            r2i_cls,
            i2r_cls,
            rgb_labels,
            ir_labels,
            rgb_idx,
            ir_idx,
            'scores',
            rgb_features,
            ir_features,
            rgb_gt,
            ir_gt
        )
        
    def _extract_feature(self, model, loader, modal):

        print('extracting {} features'.format(modal))

        saved_features, saved_labels, saved_cls= None, None, None
        saved_gts, saved_idx= None, None
        for imgs_list, infos in loader:
            labels = infos[:,1]
            idx = infos[:,0]
            gts = infos[:,-1].to(model.device)
            if imgs_list.__class__.__name__ != 'list':
                imgs = imgs_list
                imgs, labels, idx = \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device)
            else:
                ori_imgs, ca_imgs = imgs_list[0], imgs_list[1]
                if len(ori_imgs.shape) < 4:
                    ori_imgs = ori_imgs.unsqueeze(0)
                    ca_imgs = ca_imgs.unsqueeze(0)

                imgs = torch.cat((ori_imgs,ca_imgs),dim=0)
                labels = torch.cat((labels,labels),dim=0)
                idx = torch.cat((idx,idx),dim=0)
                gts= torch.cat((gts,gts),dim=0).to(model.device)
                imgs, labels, idx= \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device)
            _, bn_features = model.model(imgs) # _:gap feature

            if modal == 'rgb':
                cls, l2_features = model.classifier2(bn_features)
            elif modal == 'ir':
                cls, l2_features = model.classifier1(bn_features)
            l2_features = l2_features.detach().cpu()

            if saved_features is None: 
                # saved_features, saved_labels, saved_cls, saved_idx = l2_features, labels, cls, idx
                saved_features, saved_labels, saved_cls, saved_idx = bn_features, labels, cls, idx

                saved_gts = gts
            else:
                # saved_features = torch.cat((saved_features, l2_features), dim=0)
                saved_features = torch.cat((saved_features, bn_features), dim=0)
                saved_labels = torch.cat((saved_labels, labels), dim=0)
                saved_cls = torch.cat((saved_cls, cls), dim=0)
                saved_idx = torch.cat((saved_idx, idx), dim=0)

                saved_gts = torch.cat((saved_gts, gts), dim=0)
        return saved_features, saved_labels, saved_gts, saved_cls, saved_idx
