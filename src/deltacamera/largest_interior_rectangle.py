# Not currently used (rlemasklib has this built-in), but may come in handy later.
"""
Largest Interior Rectangle
===========================

Given a binary mask, find the largest axis-aligned rectangle that fits entirely
inside the foreground region. This is useful for finding a "clean" crop region
within a valid mask, e.g., after lens undistortion leaves an irregular boundary.

The algorithm
-------------

The naive approach would check all O(n⁴) possible rectangles, which is too slow.
Instead, we reduce the 2D problem to a series of 1D problems.

For each row, we compute a "height" array: height[j] is the number of consecutive
foreground pixels in column j, ending at the current row. If the current pixel is
background, height[j] = 0. If foreground, height[j] = height[j] + 1 from the
previous row.

This height array forms a histogram. The largest rectangle in the original mask
that has its bottom edge on this row is the largest rectangle in this histogram.
We solve this 1D problem for each row and take the maximum.

Largest rectangle in histogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 1D problem (largest rectangle in a histogram) is solved with a monotonic stack
in O(n) time. The stack maintains column indices in increasing order of height.

For each column j:
- If heights[j] >= stack top's height, push j onto the stack.
- Otherwise, pop from the stack. The popped column had a certain height h.
  The rectangle with height h extends from just after the new stack top
  (or from column 0 if stack is empty) to column j-1. Compute its area.

After processing all columns, pop remaining entries similarly, with the right
boundary at num_cols.

Complexity
----------

- Time: O(rows × cols). Each pixel is pushed and popped from the stack at most once.
- Space: O(cols) for the heights array and stack.

The algorithm is sometimes called the "histogram method" for the maximal rectangle
problem. It dates back to the 1980s.
"""

import numba
import numpy as np


@numba.njit(cache=True)
def largest_interior_rectangle(arr):
    num_rows, num_cols = arr.shape
    if num_rows == 0 or num_cols == 0:
        return np.zeros(4, dtype=np.int32)

    heights = np.zeros(num_cols, dtype=np.uint32)
    stack = np.empty(num_cols, dtype=np.uint32)
    max_area = 0
    max_height = 0
    max_bottom = 0
    max_left = 0

    for i in range(num_rows):
        for j in range(num_cols):
            if arr[i, j]:
                heights[j] += 1
            else:
                heights[j] = 0

        j = 0
        stack_idx = -1
        while j < num_cols:
            if stack_idx < 0 or heights[stack[stack_idx]] <= heights[j]:
                stack_idx += 1
                stack[stack_idx] = j
                j += 1
            else:
                top = stack[stack_idx]
                stack_idx -= 1
                if stack_idx < 0:
                    left = 0
                    area = heights[top] * j
                else:
                    left = stack[stack_idx] + 1
                    area = heights[top] * (j - left)

                if area > max_area:
                    max_area = area
                    max_bottom = i
                    max_height = heights[top]
                    max_left = left

        while stack_idx >= 0:
            top = stack[stack_idx]
            stack_idx -= 1
            if stack_idx < 0:
                left = 0
                area = heights[top] * num_cols
            else:
                left = stack[stack_idx] + 1
                area = heights[top] * (num_cols - left)
            if area > max_area:
                max_area = area
                max_bottom = i
                max_height = heights[top]
                max_left = left

    if max_height == 0:
        return np.zeros(4, dtype=np.int32)

    return np.array(
        [max_left, max_bottom + 1 - max_height, max_area // max_height, max_height], dtype=np.int32
    )


# Alternative implementation using for-loop with inner while (more standard textbook style).
# Uses >= comparison instead of <=, which affects tie-breaking when areas are equal.
@numba.njit(cache=True)
def largest_interior_rectangle2(arr):
    num_rows, num_cols = arr.shape
    if num_rows == 0 or num_cols == 0:
        return np.zeros(4, dtype=np.int32)

    heights = np.zeros(num_cols, dtype=np.uint32)
    stack = np.empty(num_cols, dtype=np.uint32)
    max_area = 0
    max_height = 0
    max_bottom = 0
    max_left = 0

    for i in range(num_rows):
        for j in range(num_cols):
            if arr[i, j]:
                heights[j] += 1
            else:
                heights[j] = 0

        j = 0
        stack_idx = -1

        for j in range(num_cols):
            while stack_idx >= 0 and heights[stack[stack_idx]] >= heights[j]:
                top = stack[stack_idx]
                stack_idx -= 1
                left = 0 if stack_idx < 0 else stack[stack_idx] + 1
                area = heights[top] * (j - left)
                if area > max_area:
                    max_area = area
                    max_bottom = i
                    max_height = heights[top]
                    max_left = left
            stack_idx += 1
            stack[stack_idx] = j
        while stack_idx >= 0:
            top = stack[stack_idx]
            stack_idx -= 1
            left = 0 if stack_idx < 0 else stack[stack_idx] + 1
            area = heights[top] * (num_cols - left)
            if area > max_area:
                max_area = area
                max_bottom = i
                max_height = heights[top]
                max_left = left

    if max_height == 0:
        return np.zeros(4, dtype=np.int32)

    return np.array(
        [max_left, max_bottom + 1 - max_height, max_area // max_height, max_height], dtype=np.int32
    )

