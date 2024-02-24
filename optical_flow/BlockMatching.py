import cv2
import numpy as np

def BlockMatching(macro_block, prev_block, BlockSize):
    # offset - shifting
    max_c = macro_block.shape[1] - prev_block.shape[1]
    max_r = macro_block.shape[0] - prev_block.shape[0]

    best_ssd = np.inf
    best_match_c = np.inf
    best_match_r = np.inf

    STEPSIZE = 1 # parameter to play around - improves performance, with precision loss

    for c_off in range(0,max_c+1, STEPSIZE):
        for r_off in range(0,max_r+1,STEPSIZE):
            m = macro_block[r_off:r_off+BlockSize, c_off:c_off+BlockSize]

            # Calculate Sum of Squared Differences (SSD) metric
            ssd = np.sum((m - prev_block) ** 2)

            # Update best match if the current SSD is smaller
            if ssd < best_ssd and ssd > 0:
                best_ssd = ssd
                best_match_c = c_off
                best_match_r = r_off

            debug = False
            if debug == True:
                large_macro_block = cv2.resize(macro_block, (448, 448), interpolation=cv2.INTER_NEAREST)
                large_prev_block = cv2.resize(prev_block, (448, 448), interpolation=cv2.INTER_NEAREST)

                macro_block_bgr = cv2.cvtColor(large_macro_block, cv2.COLOR_BAYER_BG2BGR)
                prev_block_bgr = cv2.cvtColor(large_prev_block, cv2.COLOR_BAYER_BG2BGR)

                row_scale_factor = 448//macro_block.shape[0]
                col_scale_factor = 448//macro_block.shape[1]
                scaled_frame = (r_off * row_scale_factor, c_off * col_scale_factor)

                # Draw rectangles on the frames with scaled coordinates
                rect = cv2.rectangle(macro_block_bgr.copy(), (scaled_frame[1], scaled_frame[0]),
                                        (scaled_frame[1] + BlockSize * col_scale_factor -1, scaled_frame[0] + BlockSize * row_scale_factor -1),
                                        [0, 0, 255], 1)
                rect = cv2.rectangle(rect, (0,0,), (448 -2, 448 -2),
                                        [255, 0, 255], 1)

                splot = np.hstack([rect, prev_block_bgr])
                cv2.imshow('Block Matchin - debug', splot)
                cv2.waitKey(1)

    return best_ssd, best_match_c, best_match_r