
ffmpeg -y \
  -loop 1 -t 4 -i docs/assets/sliders/upscaling1/before.png \
  -loop 1 -t 4 -i docs/assets/sliders/upscaling1/after.png \
  -filter_complex "
    [0:v]scale=800:-1,setsar=1[before];
    [1:v]scale=800:-1,setsar=1[after];
    [before][after]blend=all_expr='if(lte(X,W*(0.5-0.5*cos(2*PI*T/4))),B,A)'[blend];
    [blend]drawbox=x='W*(0.5-0.5*cos(2*PI*T/4))-2':y=0:w=4:h=ih:color=white@0.9:t=fill,fps=24,split[s0][s1];
    [s0]palettegen[p];
    [s1][p]paletteuse
  " \
  docs/assets/sliders/upscaling1/comparison.gif


ffmpeg -y   -loop 1 -t 4 -i docs/assets/sliders/upscaling2/before.jpg   -loop 1 -t 4 -i docs/assets/sliders/upscaling2/after.jpg   -filter_complex "
    [0:v]scale=800:-1,setsar=1,format=rgba[before];
    [1:v]scale=800:-1,setsar=1,format=rgba[after];
    [before][after]blend=all_expr='if(lte(X,W*(0.5-0.5*cos(2*PI*T/4))),B,A)'[blend];
    [blend]drawbox=x='w*(0.5-0.5*cos(2*PI*t/4))-2':y=0:w=4:h=ih:color=white@0.9:t=fill,fps=24,split[s0][s1];
    [s0]palettegen[p];
    [s1][p]paletteuse
  "   docs/assets/sliders/upscaling2/comparison.gif



ffmpeg -y   -loop 1 -t 4 -i docs/assets/sliders/upscaling3/before.webp   -loop 1 -t 4 -i docs/assets/sliders/upscaling3/after.webp   -filter_complex "
    [0:v]scale=800:-1,setsar=1,format=rgba[before];
    [1:v]scale=800:-1,setsar=1,format=rgba[after];
    [before][after]blend=all_expr='if(lte(X,W*(0.5-0.5*cos(2*PI*T/4))),B,A)'[blend];
    [blend]drawbox=x='w*(0.5-0.5*cos(2*PI*t/4))-2':y=0:w=4:h=ih:color=white@0.9:t=fill,fps=24,split[s0][s1];
    [s0]palettegen[p];
    [s1][p]paletteuse
  "   docs/assets/sliders/upscaling3/comparison.gif


ffmpeg -y \
  -i docs/assets/sliders/upscaling3/after.webp \
  -vf "crop=900:900:680:130,scale=800:-1" \
  docs/assets/sliders/upscaling3/after-face-preview.png

  ffmpeg -y \
  -i docs/assets/sliders/upscaling3/before.webp \
  -vf "crop=235:235:10:48,scale=800:-1" \
  docs/assets/sliders/upscaling3/before-face-preview.png


ffmpeg -y \
  -loop 1 -t 4 -i docs/assets/sliders/upscaling3/before.webp \
  -loop 1 -t 4 -i docs/assets/sliders/upscaling3/after.webp \
  -filter_complex "
    [0:v]crop=220:220:179:54,scale=800:-1,setsar=1,format=rgba[before];
    [1:v]crop=900:900:680:130,scale=800:-1,setsar=1,format=rgba[after];
    [before][after]blend=all_expr='if(lte(X,W*(0.5-0.5*cos(2*PI*T/4))),B,A)'[blend];
    [blend]drawbox=x='w*(0.5-0.5*cos(2*PI*t/4))-2':y=0:w=4:h=ih:color=white@0.9:t=fill,fps=24,split[s0][s1];
    [s0]palettegen[p];
    [s1][p]paletteuse
  " \
  docs/assets/sliders/upscaling3/comparison-face.gif


ffmpeg -y   -loop 1 -t 4 -i docs/assets/sliders/upscaling3/before-face-preview.png   -loop 1 -t 4 -i docs/assets/sliders/upscaling3/after-face-preview.png   -filter_complex "
    [0:v]scale=800:-1,setsar=1,format=rgba[before];
    [1:v]scale=800:-1,setsar=1,format=rgba[after];
    [before][after]blend=all_expr='if(lte(X,W*(0.5-0.5*cos(2*PI*T/4))),B,A)'[blend];
    [blend]drawbox=x='w*(0.5-0.5*cos(2*PI*t/4))-2':y=0:w=4:h=ih:color=white@0.9:t=fill,fps=24,split[s0][s1];
    [s0]palettegen[p];
    [s1][p]paletteuse
  "   docs/assets/sliders/upscaling3/comparison-p.gif

