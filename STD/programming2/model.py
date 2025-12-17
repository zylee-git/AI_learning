import math

import torch
import torch.nn as nn
import torchvision
from torchvision import models

from utils import compute_offsets, assign_label, generate_proposal
from loss import ClsScoreRegression, BboxRegression


class FeatureExtractor(nn.Module):
    """
    Image feature extraction with MobileNet.
    """
    def __init__(self, reshape_size=224, pooling=False, verbose=False):
        super().__init__()

        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1]) # Remove the last classifier

        # average pooling
        if pooling:
            self.mobilenet.add_module('LastAvgPool', nn.AvgPool2d(math.ceil(reshape_size/32.))) # input: N x 1280 x 7 x 7

        for i in self.mobilenet.named_parameters():
            i[1].requires_grad = True # fine-tune all

    def forward(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape Nx3x224x224

        Outputs:
        - feat: Image feature, of shape Nx1280 (pooled) or Nx1280x7x7
        """
        num_img = img.shape[0]

        img_prepro = img

        feat = []
        process_batch = 500
        for b in range(math.ceil(num_img/process_batch)):
            feat.append(self.mobilenet(img_prepro[b*process_batch:(b+1)*process_batch]
                                    ).squeeze(-1).squeeze(-1)) # forward and squeeze
        feat = torch.cat(feat)

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat


class FastRCNN(nn.Module):
    def __init__(self, in_dim=1280, hidden_dim=256, num_classes=20, \
                roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
        super().__init__()

        assert(num_classes != 0)
        self.num_classes = num_classes
        self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
        self.feat_extractor = FeatureExtractor()
        ##############################################################################
        # TODO: Declare the cls & bbox heads (in Fast R-CNN).                        #
        # The cls & bbox heads share a sequential module with a Linear layer,        #
        # followed by a Dropout (p=drop_ratio), a ReLU nonlinearity and another      #
        # Linear layer.                                                              #
        # The cls head is a Linear layer that predicts num_classes + 1 (background). #
        # The bbox head is a Linear layer that predicts offsets(dim=4).              #
        # HINT: The dimension of the two Linear layers are in_dim -> hidden_dim and  #
        # hidden_dim -> hidden_dim.                                                  #
        ##############################################################################
        # Replace "pass" statement with your code
        self.shared_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(drop_ratio),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(hidden_dim, 4) 
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, images, bboxes, bbox_batch_ids, proposals, proposal_batch_ids):
        """
        Training-time forward pass for our two-stage Faster R-CNN detector.

        Inputs:
        - images: Tensor of shape (B, 3, H, W) giving input images
        - bboxes: Tensor of shape (N, 5) giving ground-truth bounding boxes
        and category labels, from the dataloader, where N is the total number
        of GT boxes in the batch
        - bbox_batch_ids: Tensor of shape (N, ) giving the index (in the batch)
        of the image that each GT box belongs to
        - proposals: Tensor of shape (M, 4) giving the proposals for input images, 
        where M is the total number of proposals in the batch
        - proposal_batch_ids: Tensor of shape (M, ) giving the index of the image 
        that each proposals belongs to

        Outputs:
        - total_loss: Torch scalar giving the overall training loss.
        """
        w_cls = 1 # for cls_scores
        w_bbox = 1 # for offsets
        total_loss = None
        ##############################################################################
        # TODO: Implement the forward pass of Fast R-CNN.                            #
        # A few key steps are outlined as follows:                                   #
        # i) Extract image feature.                                                  #
        # ii) Perform RoI Pool on proposals and meanpool the feature in the spatial  #
        #     dimension.                                                             #
        # iii) Pass the RoI feature through the shared-fc layer. Predict             #
        #      classification scores and box offsets                                 #
        # iv) Assign the proposals with targets of each image.                       #
        #     HINT: Use `assign_label` in utils.py.                                  #
        #     NOTE: foreground ids are in [0, self.num_classes-1], and background id #
        #           is self.num_classes. (see dataset.py)                            #
        # v) Compute the cls_loss between the predicted class_prob and GT_class      #
        #    (For poistive & negative proposals)                                     #
        #    Compute the bbox_loss between the offsets and GT_offsets                #
        #    (For positive proposals)                                                #
        #    Compute the total_loss which is formulated as:                          #
        #    total_loss = w_cls*cls_loss + w_bbox*bbox_loss.                         #
        ##############################################################################
        # Replace "pass" statement with your code
        B, _, H, W = images.shape
        
        # extract image feature
        img_features = self.feat_extractor(images)

        # perform RoI Pool & mean pool
        rois = torch.cat([proposal_batch_ids.unsqueeze(1).float(), proposals], dim=1)
        roi_features = torchvision.ops.roi_pool(img_features, rois, (self.roi_output_h, self.roi_output_w), spatial_scale=1.0/32.0)
        roi_features = roi_features.mean(dim=(2, 3))

        # forward heads, get predicted cls scores & offsets
        shared_features = self.shared_fc(roi_features)
        cls_scores = self.cls_head(shared_features)
        offsets = self.bbox_head(shared_features)

        # assign targets with proposals
        pos_masks, neg_masks, GT_labels, GT_bboxes = [], [], [], []
        cls_loss = 0
        bbox_loss = 0
        batch_size = B
        for img_idx in range(B):
            # get the positive/negative proposals and corresponding
            # GT box & class label of this image
            img_proposals_mask = (proposal_batch_ids == img_idx)
            img_bboxes_mask = (bbox_batch_ids == img_idx)
            
            if img_proposals_mask.sum() == 0 or img_bboxes_mask.sum() == 0:
                continue
                
            img_proposals = proposals[img_proposals_mask]
            img_bboxes = bboxes[img_bboxes_mask]
            
            # Assign labels
            pos_mask, neg_mask, GT_labels, GT_bboxes = assign_label(
                img_proposals, img_bboxes, self.num_classes)
            
            # Get predictions for this image
            img_cls_scores = cls_scores[img_proposals_mask]
            img_offsets = offsets[img_proposals_mask]
            
            # Classification loss (positive + negative proposals)
            pos_neg_mask = pos_mask | neg_mask
            if pos_neg_mask.sum() > 0:
                cls_loss += ClsScoreRegression(
                    img_cls_scores[pos_neg_mask], 
                    GT_labels[pos_neg_mask], 
                    batch_size)
            
            # Bbox regression loss (only positive proposals)
            if pos_mask.sum() > 0:
                # Compute GT offsets for positive proposals
                pos_proposals = img_proposals[pos_mask]
                pos_GT_bboxes = GT_bboxes
                GT_offsets = compute_offsets(pos_proposals, pos_GT_bboxes)
                
                bbox_loss += BboxRegression(
                    img_offsets[pos_mask], 
                    GT_offsets, 
                    batch_size)

        # compute loss
        total_loss = w_cls * cls_loss + w_bbox * bbox_loss
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return total_loss

    def inference(self, images, proposals, proposal_batch_ids, thresh=0.5, nms_thresh=0.7):
        """"
        Inference-time forward pass for our two-stage Faster R-CNN detector

        Inputs:
        - images: Tensor of shape (B, 3, H, W) giving input images
        - proposals: Tensor of shape (M, 4) giving the proposals for input images, 
        where M is the total number of proposals in the batch
        - proposal_batch_ids: Tensor of shape (M, ) giving the index of the image 
        that each proposals belongs to
        - thresh: Threshold value on confidence probability. HINT: You can convert the
        classification score to probability using a softmax nonlinearity.
        - nms_thresh: IoU threshold for NMS

        We can output a variable number of predicted boxes per input image.
        In particular we assume that the input images[i] gives rise to P_i final
        predicted boxes.

        Outputs:
        - final_proposals: List of length (B,) where final_proposals[i] is a Tensor
        of shape (P_i, 4) giving the coordinates of the final predicted boxes for
        the input images[i]
        - final_conf_probs: List of length (B,) where final_conf_probs[i] is a
        Tensor of shape (P_i, 1) giving the predicted probabilites that the boxes
        in final_proposals[i] are objects (vs background)
        - final_class: List of length (B,), where final_class[i] is an int64 Tensor
        of shape (P_i, 1) giving the predicted category labels for each box in
        final_proposals[i].
        """
        final_proposals, final_conf_probs, final_class = None, None, None
        ##############################################################################
        # TODO: Predicting the final proposal coordinates `final_proposals`,         #
        # confidence scores `final_conf_probs`, and the class index `final_class`.   #
        # The overall steps are similar to the forward pass, but now you cannot      #
        # decide the activated nor negative proposals without GT boxes.              #
        # You should apply post-processing (thresholding and NMS) to all proposals.  #
        # A few key steps are outlined as follows:                                   #
        # i) Extract image feature.                                                  #
        # ii) Perform RoI Pooling on proposals, then apply mean pooling on the RoI   #
        #     features in the spatial dimensions.                                    #
        #     HINT: Use torchvision.ops.roi_pool.                                    #
        # iii) Pass the RoI features through the shared-fc layer. Predict            #
        #      classification scores and box offsets.                                #
        # iv) Obtain the predicted class, confidence score, and bounding box for     #
        #     each proposal. Note that the confidence score is the probability of    #
        #     the predicted class (apply softmax to classification scores first).    #
        #     HINT: Use generate_proposal from utils.py to get predicted boxes from  #
        #     offsets.                                                               #
        # v) Apply post-processing to filter and obtain the final predictions for    #
        #    each image in the batch:                                                #
        #    - Thresholding: Filter out predictions with confidence lower than       #
        #      threshold.                                                            #
        #    - Non-Maximum Suppression (NMS): Apply NMS to the remaining predictions #
        #      for each image (do not distinguish between different classes).        #
        #      HINT: Use torchvision.ops.nms.                                        #
        ##############################################################################
        # Replace "pass" statement with your code
        B = images.shape[0]

        # extract image feature
        img_features = self.feat_extractor(images)

        # perform RoI Pool & mean pool
        rois = torch.cat([proposal_batch_ids.unsqueeze(1).float(), proposals], dim=1)
        roi_features = torchvision.ops.roi_pool(img_features, rois, (self.roi_output_h, self.roi_output_w), spatial_scale=1.0/32.0)
        roi_features = roi_features.mean(dim=(2, 3))

        # forward heads, get predicted cls scores & offsets
        shared_features = self.shared_fc(roi_features)
        cls_scores = self.cls_head(shared_features)
        offsets = self.bbox_head(shared_features)
        cls_probs = F.softmax(cls_scores, dim=-1)

        # get predicted boxes & class label & confidence probability
        conf_probs, pred_class = cls_probs[:, :-1].max(dim=-1)
        pred_class = pred_class  # 0-19 for object classes
        pred_boxes = generate_proposal(proposals, offsets)

        final_proposals = []
        final_conf_probs = []
        final_class = []
        # post-process to get final predictions
        for img_idx in range(B):

            # filter by threshold
            img_mask = (proposal_batch_ids == img_idx)
        
            if img_mask.sum() == 0:
                final_proposals.append(torch.empty(0, 4, device=proposals.device))
                final_conf_probs.append(torch.empty(0, 1, device=proposals.device))
                final_class.append(torch.empty(0, 1, device=proposals.device, dtype=torch.long))
                continue
            
            img_boxes = pred_boxes[img_mask]
            img_conf = conf_probs[img_mask]
            img_class = pred_class[img_mask]
            
            keep_thresh = img_conf > thresh
            img_boxes = img_boxes[keep_thresh]
            img_conf = img_conf[keep_thresh]
            img_class = img_class[keep_thresh]
            
            if len(img_boxes) == 0:
                final_proposals.append(torch.empty(0, 4, device=proposals.device))
                final_conf_probs.append(torch.empty(0, 1, device=proposals.device))
                final_class.append(torch.empty(0, 1, device=proposals.device, dtype=torch.long))
                continue

            # nms
            keep_nms = torchvision.ops.nms(img_boxes, img_conf, nms_thresh)
        
            final_proposals.append(img_boxes[keep_nms])
            final_conf_probs.append(img_conf[keep_nms].unsqueeze(1))
            final_class.append(img_class[keep_nms].unsqueeze(1))

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return final_proposals, final_conf_probs, final_class