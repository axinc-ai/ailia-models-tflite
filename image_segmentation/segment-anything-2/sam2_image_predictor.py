import sys
import os
import time

import numpy as np
import cv2

import os
import numpy as np
from typing import Optional
from typing import Tuple

from model_utils import format_input_tensor, get_output_tensor

class SAM2ImagePredictor:
    debug = False
    dump = False

    def __init__(
        self,
        image_size,
        debug,
        dump,
        accuracy
    ):
        self.image_size = image_size
        self.debug = debug
        self.dump = dump
        self.accuracy = accuracy

    def dump_tensor(self, path, tensor):
        if type(tensor) == bool:
            if tensor:
                data = np.array([1], dtype=np.float32)
            else:
                data = np.array([0], dtype=np.float32)
        else:
            data = tensor.flatten()
        import struct
        s = struct.pack('f'*len(data), *data)
        f = open(path,'wb')
        f.write(s)
        f.close()

    def trunc_normal(self, size, std=0.02, a=-2, b=2):
        values = np.random.normal(loc=0., scale=std, size=size)
        values = np.clip(values, a*std, b*std)       
        return values

    def set_image(self, image, image_encoder):
        img = np.expand_dims(image, 0)
        img = img.astype(np.float32)
        #print(img.shape)

        #image_encoder.allocate_tensors()
        input_details = image_encoder.get_input_details()
        output_details = image_encoder.get_output_details()
        #interpreter.resize_tensor_input(
        #    input_details[0]["index"], 
        #    [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]
        #)
        #image_encoder.allocate_tensors()

        if self.accuracy == "int8":
            image_encoder.set_tensor(input_details[0]["index"], format_input_tensor(img, input_details, 0))
            image_encoder.invoke()
            vision_features = get_output_tensor(image_encoder, output_details, 4)
            vision_pos_enc_0 = get_output_tensor(image_encoder, output_details, 1)
            vision_pos_enc_1 = get_output_tensor(image_encoder, output_details, 5)
            vision_pos_enc_2 = get_output_tensor(image_encoder, output_details, 3)
            backbone_fpn_0 = get_output_tensor(image_encoder, output_details, 0)
            backbone_fpn_1 = get_output_tensor(image_encoder, output_details, 2)
            backbone_fpn_2 = get_output_tensor(image_encoder, output_details, 6)
        else:
            image_encoder.set_tensor(input_details[0]["index"], img)
            image_encoder.invoke()

            vision_features = image_encoder.get_tensor(output_details[4]["index"]) # 4 or 6
            vision_pos_enc_0 = image_encoder.get_tensor(output_details[1]["index"])
            vision_pos_enc_1 = image_encoder.get_tensor(output_details[5]["index"])
            vision_pos_enc_2 = image_encoder.get_tensor(output_details[3]["index"])
            backbone_fpn_0 = image_encoder.get_tensor(output_details[0]["index"])
            backbone_fpn_1 = image_encoder.get_tensor(output_details[2]["index"])
            backbone_fpn_2 = image_encoder.get_tensor(output_details[6]["index"])

        if self.dump:
            self.dump_tensor("image_encoder_input_0.dat", img)
            self.dump_tensor("image_encoder_output_0.dat", backbone_fpn_0)
            self.dump_tensor("image_encoder_output_1.dat", vision_pos_enc_0)
            self.dump_tensor("image_encoder_output_2.dat", backbone_fpn_1)
            self.dump_tensor("image_encoder_output_3.dat", vision_pos_enc_2)
            self.dump_tensor("image_encoder_output_4.dat", vision_features)
            self.dump_tensor("image_encoder_output_5.dat", vision_pos_enc_1)
            self.dump_tensor("image_encoder_output_6.dat", backbone_fpn_2)

        if self.debug:
            print("vision_features", vision_features.shape)
            print("vision_pos_enc_0", vision_pos_enc_0.shape)
            print("vision_pos_enc_1", vision_pos_enc_1.shape)
            print("vision_pos_enc_2", vision_pos_enc_2.shape)
            print("backbone_fpn_0", backbone_fpn_0.shape)
            print("backbone_fpn_1", backbone_fpn_1.shape)
            print("backbone_fpn_2", backbone_fpn_2.shape)

        backbone_out = {"vision_features":vision_features,
                        "vision_pos_enc":[vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2],
                        "backbone_fpn":[backbone_fpn_0,backbone_fpn_1, backbone_fpn_2]}

        _, vision_feats, _, _ = self._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        directly_add_no_mem_embed = True
        if directly_add_no_mem_embed:
            hidden_dim = 256
            no_mem_embed = self.trunc_normal((1, 1, hidden_dim), std=0.02).astype(np.float32)
            vision_feats[-1] = vision_feats[-1] + no_mem_embed

        bb_feat_sizes = [
            (self.image_size // 4, self.image_size // 4),
            (self.image_size // 8, self.image_size // 8),
            (self.image_size // 16, self.image_size // 16),
        ]

        feats = [
            np.transpose(feat, (1, 2, 0)).reshape(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]

        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return features

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        num_feature_levels = 3

        feature_maps = backbone_out["backbone_fpn"][-num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [np.transpose(x.reshape(x.shape[0], x.shape[1], -1), (2, 0, 1)) for x in feature_maps]
        vision_pos_embeds = [np.transpose(x.reshape(x.shape[0], x.shape[1], -1), (2, 0, 1)) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def predict(
        self,
        features,
        orig_hw,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
        prompt_encoder = None,
        mask_decoder = None,
        onnx = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
        point_coords, point_labels, box, mask_input, normalize_coords, orig_hw
        )

        masks, iou_predictions, low_res_masks = self._predict(
            features,
            orig_hw,
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            onnx=onnx
        )

        return masks[0], iou_predictions[0], low_res_masks[0]

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, orig_hw
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            point_coords = point_coords.astype(np.float32)
            unnorm_coords = self.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=orig_hw
            )
            labels = point_labels.astype(np.int64)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = box.astype(np.float32)
            unnorm_box = self.transform_boxes(
                box, normalize=normalize_coords, orig_hw=orig_hw
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = mask_input.astype(np.float32)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(
        self, 
        features,
        orig_hw,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        boxes: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        prompt_encoder = None,
        mask_decoder = None,
        onnx = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if point_coords is not None:
            concat_points = (point_coords, point_labels.astype(np.int32))
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = np.array([[2, 3]])
            box_labels = box_labels.repeat(boxes.shape[0], 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = np.concatenate([box_coords, concat_points[0]], axis=1)
                concat_labels = np.concatenate([box_labels, concat_points[1]], axis=1)
                concat_points = (concat_coords, concat_labels.astype(np.int32))
            else:
                concat_points = (box_coords, box_labels.astype(np.int32))

        if mask_input is None:
            mask_input_dummy = np.zeros((1, self.image_size // 4, self.image_size // 4), dtype=np.float32)
            masks_enable = np.array([0], dtype=np.int32)
        else:
            mask_input_dummy = mask_input
            masks_enable = np.array([1], dtype=np.int32)

        if concat_points is None:
            raise("concat_points must be exists")

        #prompt_encoder.allocate_tensors()
        input_details = prompt_encoder.get_input_details()
        output_details = prompt_encoder.get_output_details()

        #padding
        padding_length = input_details[2]["shape"][1]
        original_length = concat_points[0].shape[1]
        concat_points_pad = (
            np.zeros((concat_points[0].shape[0], padding_length, concat_points[0].shape[2]), dtype=np.float32),
            -np.ones((concat_points[1].shape[0], padding_length), dtype=np.float32)
        )
        concat_points_pad[0][:, 0:concat_points[0].shape[1], :] = concat_points[0]
        concat_points_pad[1][:, 0:concat_points[1].shape[1]] = concat_points[1]
        concat_points = concat_points_pad

        if self.accuracy == "int8" or self.accuracy == "mixed":
            prompt_encoder.set_tensor(input_details[2]["index"], format_input_tensor(concat_points[0], input_details, 2))
            prompt_encoder.set_tensor(input_details[3]["index"], format_input_tensor(concat_points[1], input_details, 3))
            prompt_encoder.set_tensor(input_details[0]["index"], format_input_tensor(mask_input_dummy, input_details, 0))
            prompt_encoder.set_tensor(input_details[1]["index"], format_input_tensor(masks_enable, input_details, 1))
            prompt_encoder.invoke()

            sparse_embeddings = get_output_tensor(prompt_encoder, output_details, 1)
            dense_embeddings = get_output_tensor(prompt_encoder, output_details, 0)
            dense_pe = get_output_tensor(prompt_encoder, output_details, 2)
        else:
            prompt_encoder.set_tensor(input_details[2]["index"], concat_points[0])
            prompt_encoder.set_tensor(input_details[3]["index"], concat_points[1])
            prompt_encoder.set_tensor(input_details[0]["index"], mask_input_dummy)
            prompt_encoder.set_tensor(input_details[1]["index"], masks_enable)
            prompt_encoder.invoke()

            sparse_embeddings = prompt_encoder.get_tensor(output_details[1]["index"])
            dense_embeddings = prompt_encoder.get_tensor(output_details[0]["index"])
            dense_pe = prompt_encoder.get_tensor(output_details[2]["index"])

        if self.dump:
            self.dump_tensor("prompt_encoder_input_2.dat", concat_points[0])
            self.dump_tensor("prompt_encoder_input_3.dat", concat_points[1])
            self.dump_tensor("prompt_encoder_input_0.dat", mask_input_dummy)
            self.dump_tensor("prompt_encoder_input_1.dat", masks_enable)

            self.dump_tensor("prompt_encoder_output_1.dat", sparse_embeddings)
            self.dump_tensor("prompt_encoder_output_0.dat", dense_embeddings)
            self.dump_tensor("prompt_encoder_output_2.dat", dense_pe)

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level
            for feat_level in features["high_res_feats"]
        ]

        image_feature = features["image_embed"]

        if self.debug:
            print("coords", concat_points[0].shape)
            print("labels", concat_points[1].shape)
            print("masks", mask_input_dummy.shape)

            print("sparse_embeddings", sparse_embeddings.shape)
            print("dense_embeddings", dense_embeddings.shape)
            print("dense_pe", dense_pe.shape)

        #mask_decoder.allocate_tensors()
        input_details = mask_decoder.get_input_details()
        output_details = mask_decoder.get_output_details()

        #batched_mode_np = np.zeros((1), dtype=bool)
        #if batched_mode:
        #    batched_mode_np[0] = True
        
        if self.debug:
            print("high_res_features[0]", high_res_features[0].shape)
            print("high_res_features[1]", high_res_features[1].shape)
        
        sparse_embeddings = sparse_embeddings[:,:original_length + 1,:]

        padding_length = input_details[1]["shape"][1]
        sparse_embeddings_pad = np.zeros((sparse_embeddings.shape[0], padding_length, sparse_embeddings.shape[2]), np.float32)
        sparse_embeddings_pad[:, 0:sparse_embeddings.shape[1], :] = sparse_embeddings
        sparse_embeddings = sparse_embeddings_pad

        attn_masks = np.zeros((sparse_embeddings.shape[0], padding_length), dtype = np.bool_)
        attn_masks[:, 0:original_length + 1] = True

        if self.accuracy == "int8" or self.accuracy == "mixed":
            mask_decoder.set_tensor(input_details[3]["index"], format_input_tensor(image_feature, input_details, 3))
            mask_decoder.set_tensor(input_details[6]["index"], format_input_tensor(dense_pe, input_details, 6))
            mask_decoder.set_tensor(input_details[1]["index"], format_input_tensor(sparse_embeddings, input_details, 1))
            mask_decoder.set_tensor(input_details[2]["index"], format_input_tensor(dense_embeddings, input_details, 2))
            mask_decoder.set_tensor(input_details[5]["index"], batched_mode)
            mask_decoder.set_tensor(input_details[0]["index"], format_input_tensor(high_res_features[0], input_details, 0))
            mask_decoder.set_tensor(input_details[4]["index"], format_input_tensor(high_res_features[1], input_details, 4))
            if len(input_details) >= 8:
                mask_decoder.set_tensor(input_details[7]["index"], attn_masks)
            mask_decoder.invoke()

            masks = get_output_tensor(mask_decoder, output_details, 2)
            iou_pred = get_output_tensor(mask_decoder, output_details, 0)
            sam_tokens_out = get_output_tensor(mask_decoder, output_details, 3)
            object_score_logits = get_output_tensor(mask_decoder, output_details, 1)
        else:
            mask_decoder.set_tensor(input_details[3]["index"], image_feature)
            mask_decoder.set_tensor(input_details[6]["index"], dense_pe)
            mask_decoder.set_tensor(input_details[1]["index"], sparse_embeddings)
            mask_decoder.set_tensor(input_details[2]["index"], dense_embeddings)
            mask_decoder.set_tensor(input_details[5]["index"], batched_mode)
            mask_decoder.set_tensor(input_details[0]["index"], high_res_features[0])
            mask_decoder.set_tensor(input_details[4]["index"], high_res_features[1])
            if len(input_details) >= 8:
                mask_decoder.set_tensor(input_details[7]["index"], attn_masks)
            mask_decoder.invoke()

            masks = mask_decoder.get_tensor(output_details[2]["index"])
            iou_pred = mask_decoder.get_tensor(output_details[0]["index"])
            sam_tokens_out = mask_decoder.get_tensor(output_details[3]["index"])
            object_score_logits = mask_decoder.get_tensor(output_details[1]["index"])

        if self.dump:
            self.dump_tensor("mask_decoder_input_3.dat", image_feature)
            self.dump_tensor("mask_decoder_input_6.dat", dense_pe)
            self.dump_tensor("mask_decoder_input_1.dat", sparse_embeddings)
            self.dump_tensor("mask_decoder_input_2.dat", dense_embeddings)
            self.dump_tensor("mask_decoder_input_5.dat", batched_mode)
            self.dump_tensor("mask_decoder_input_0.dat", high_res_features[0])
            self.dump_tensor("mask_decoder_input_4.dat", high_res_features[1])
            if len(input_details) >= 8:
                self.dump_tensor("mask_decoder_input_7.dat", attn_masks)

            self.dump_tensor("mask_decoder_output_2.dat", masks)
            self.dump_tensor("mask_decoder_output_0.dat", iou_pred)
            self.dump_tensor("mask_decoder_output_3.dat", sam_tokens_out)
            self.dump_tensor("mask_decoder_output_1.dat", object_score_logits)

        if self.debug:
            print("masks", masks.shape)
            print("iou_pred", iou_pred.shape)
            print("sam_tokens_out", sam_tokens_out.shape)
            print("object_score_logits", object_score_logits.shape)

        low_res_masks, iou_predictions, _, _  = self.forward_postprocess(masks, iou_pred, sam_tokens_out, object_score_logits, multimask_output)

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(
            low_res_masks, orig_hw
        )
        low_res_masks = np.clip(low_res_masks, -32.0, 32.0)
        mask_threshold = 0.0
        if not return_logits:
            masks = masks > mask_threshold

        return masks, iou_predictions, low_res_masks

    def forward_postprocess(
        self,
        masks,
        iou_pred,
        mask_tokens_out,
        object_score_logits,
        multimask_output: bool,
    ):
        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        #elif self.dynamic_multimask_via_stability and not self.training:
        #    masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        use_multimask_token_for_obj_ptr = True
        if multimask_output and use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def transform_coords(
        self, coords, normalize=False, orig_hw=None
    ):
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.copy()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        resolution = self.image_size
        coords = coords * resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes, normalize=False, orig_hw=None
    ):
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: np.ndarray, orig_hw) -> np.ndarray:
        interpolated_masks = []
        for mask in masks:
            mask = np.transpose(mask, (1, 2, 0))
            resized_mask = cv2.resize(mask, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
            resized_mask = np.transpose(resized_mask, (2, 0, 1))
            interpolated_masks.append(resized_mask)
        interpolated_masks = np.array(interpolated_masks)

        return interpolated_masks
