from pathlib import Path

import cv2
import pandas as pd
from matplotlib import pyplot as plt

from playground.SmartRandomCropAugmentation import PointTileAugmentation, visualize_augmentation_results

if __name__ == "__main__":
    """
    images,x,y,labels
    L_07_05_16_DSC00126.JPG,2504.0,392.0,3
    L_07_05_16_DSC00126.JPG,1726.5,458.5,3
    L_07_05_16_DSC00127.JPG,2626.5,2337.5,3
    L_07_05_16_DSC00127.JPG,1818.0,2462.5,3
    L_07_05_16_DSC00150.JPG,1539.0,1411.0,2
    L_07_05_16_DSC00150.JPG,1121.0,1377.5,2
    L_07_05_16_DSC00150.JPG,939.5,1413.5,2
    """
    labels_path = Path("/home/christian/hnee/HerdNet/data/train.csv")



    images_path = Path("/home/christian/hnee/HerdNet/data/train")

    df_labels = pd.read_csv(labels_path)

    # Create augmenter
    augmenter = PointTileAugmentation(
        tile_size=512,
        empty_prob=0.2,  # 20% empty tiles
        rotation_limit=275.0,
        min_points_in_tile=1
    )

    results = []
    # Get points for specific image
    image_points = []
    for i, (image_name, row) in enumerate(df_labels.groupby('images')):
        image = cv2.imread(images_path / image_name)
        for _, label_row in row.iterrows():
            image_points.append((label_row['x'], label_row['y'], label_row['labels']))

        print(f"Processing image: {image_name} with {len(image_points)} points")
        # for i in range(10):
        #     tile, transformed_points, is_empty = augmenter.augment_tile(image, image_points)
        #     results.append({
        #         'tile_id': i,
        #         'num_points': len(transformed_points),
        #         'is_empty': is_empty,
        #         'tile_shape': tile.shape
        #     })
        #
        #     if i < 3:  # Show details for first 3 tiles
        #         print(f"\nTile {i + 1}:")
        #         print(f"  Points in tile: {len(transformed_points)}")
        #         print(f"  Is empty: {is_empty}")
        #         print(f"  Point coordinates: {[(round(x, 1), round(y, 1), l) for x, y, l in transformed_points]}")
        #
        # # Summary statistics
        # empty_tiles = sum(1 for r in results if r['is_empty'])
        # print(f"\nSummary (10 tiles):")
        # print(f"  Empty tiles: {empty_tiles}/10 ({empty_tiles * 10}%)")
        # print(f"  Non-empty tiles: {10 - empty_tiles}/10")

        # Create visualization
        fig = visualize_augmentation_results(image, image_points, augmenter, num_examples=6)
        plt.show()

        if i > 10:
            break