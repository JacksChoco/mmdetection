_base_ = './htc-without-semantic_r50_fpn_1x_coco.py'

data_root = 'Dentex2-2'

model = dict(
    data_preprocessor=dict(pad_seg=True),
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=None
        # semantic_head=dict(
        #     type='FusedSemanticHead',
        #     num_ins=5,
        #     fusion_level=1,
        #     seg_scale_factor=1 / 8,
        #     num_convs=4,
        #     in_channels=256,
        #     conv_out_channels=256,
        #     num_classes=183,
        #     loss_seg=dict(
        #         type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))
        #         ))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=False),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img='train'),
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img='valid'),
        data_root=data_root,
        ann_file='valid/_annotations.coco.json'
        ))
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img='test'),
        data_root=data_root,
        ann_file='test/_annotations.coco.json'
        ))

val_evaluator = dict(
    ann_file= data_root + '/valid/_annotations.coco.json'
)
test_evaluator = dict(
    ann_file= data_root + '/test/_annotations.coco.json'
)