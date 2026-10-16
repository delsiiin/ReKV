"""Spatial anchors of OneVision video tokens, in normalized source coordinates.

The vision encoder mixes information across patches. These boxes describe the
patch lattice and bilinear pooling geometry, not exclusive receptive fields.
"""
import math


def _axis_weights(index, inputs, outputs):
    # torch.interpolate(..., mode='bilinear', align_corners=False).
    coordinate = min(inputs - 1, max(0.0, (index + 0.5) * inputs / outputs - 0.5))
    low = math.floor(coordinate)
    high = min(inputs - 1, low + 1)
    weight = coordinate - low
    return [(low, 1.0 - weight)] + ([(high, weight)] if high != low and weight > 0 else [])


def video_token_layout(processor, vision_config, processed_shape, tokens_per_frame):
    """Map the supported direct-resize video pipeline back to each original frame."""
    height, width = map(int, processed_shape)
    size = processor.size
    if getattr(processor, 'do_center_crop', False):
        raise ValueError('Spatial mapping does not support center-cropped video processors')
    if not processor.do_resize or size.get('height') != height or size.get('width') != width:
        raise ValueError('Spatial mapping requires the OneVision explicit height/width resize')
    patch = vision_config.patch_size
    patch_rows, patch_cols = height // patch, width // patch
    rows, cols = math.ceil(patch_rows / 2), math.ceil(patch_cols / 2)
    if height != vision_config.image_size or width != vision_config.image_size:
        raise ValueError('Processed video dimensions do not match the vision patch lattice')
    if rows * cols != tokens_per_frame:
        raise ValueError('Video token count does not match the bilinear pooled patch lattice')
    cells = []
    for row in range(rows):
        for col in range(cols):
            patches = [dict(row=y, col=x, index=y * patch_cols + x, weight=wy * wx)
                       for y, wy in _axis_weights(row, patch_rows, rows)
                       for x, wx in _axis_weights(col, patch_cols, cols)]
            cells.append(dict(
                index=row * cols + col, row=row, col=col,
                # Disjoint cells centred on the bilinear sampling locations.
                box=[col * patch_cols * patch / cols / width,
                     row * patch_rows * patch / rows / height,
                     (col + 1) * patch_cols * patch / cols / width,
                     (row + 1) * patch_rows * patch / rows / height],
                support_box=[min(p['col'] for p in patches) * patch / width,
                             min(p['row'] for p in patches) * patch / height,
                             (max(p['col'] for p in patches) + 1) * patch / width,
                             (max(p['row'] for p in patches) + 1) * patch / height],
                patches=patches,
            ))
    return dict(rows=rows, cols=cols, patch_rows=patch_rows, patch_cols=patch_cols,
                patch_size=patch, processed_width=width, processed_height=height,
                covered_box=[0, 0, patch_cols * patch / width, patch_rows * patch / height],
                transform='direct resize; no center crop or padding in video preprocessing',
                pooling='bilinear, align_corners=False', coordinate_system='normalized source image [x0,y0,x1,y1]',
                interpretation='Spatial anchors and pooling support, not exclusive receptive fields', cells=cells)
