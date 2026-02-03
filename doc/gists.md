```shell

rsync -av  --progress --stats --human-readable --include="*.png" --include="*.PNG" --exclude="*" -e "ssh -i .ssh/id_ed25519" "cwinkelmann@10.188.1.1:/raid/cwinkelmann/herdnet/outputs/2025-09-10/17-55-19/visualizations/" '/Users/christian/data/Iguanas_From_Above/visualisations' 

rsync -av  --progress --stats --human-readable --exclude="*" --include="*.png" --include="*.PNG"  -e "ssh -i .ssh/id_ed25519" "cwinkelmann@10.188.1.1:/raid/cwinkelmann/herdnet/outputs/2025-09-10" '/Users/christian/data/Iguanas_From_Above/visualisations' 

rsync -av  --progress --stats --human-readable -e "ssh -i .ssh/id_ed25519" "cwinkelmann@10.188.1.1:/storage/cwinkelmann/Iguanas_From_Above/2020_2021_2022_2023_2024/Fernandina_processed/output" '/Volumes/G-DRIVE/Metashape_Orthomosaics/Fernandina_processed/'


rsync -av  --progress --stats --human-readable -e "ssh -i .ssh/id_ed25519" "cwinkelmann@10.188.1.1:/home/cwinkelmann/work/Herdnet/best_models/17-02-44_convnext_camouflaged_zr7ljum7/best_model.pth" '/Users/christian/PycharmProjects/hnee/HerdNet/best_models/17-02-44_convnext_camouflaged_zr7ljum7'

rsync -av  --progress --stats --human-readable -e "ssh -i .ssh/id_ed25519" "cwinkelmann@10.188.1.1:/home/cwinkelmann/work/Herdnet/best_models/10-51-48_20251216_x25_dla34_ys2uq6yf/best_model.pth" '/Users/christian/PycharmProjects/hnee/HerdNet/best_models/10-51-48_20251216_x25_dla34_ys2uq6yf/best_model.pth'

```


```yaml
ffmpeg -framerate 1.5 -pattern_type glob -i "/Users/christian/data/Iguanas_From_Above/visualisations/visualizations/gif_folder/*.png" -vf "palettegen" palette.png


magick convert -delay 67 -loop 0 $(ls /Users/christian/data/Iguanas_From_Above/visualisations/visualizations/gif_folder/*.png | sort -V) output.gif
```
