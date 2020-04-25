set p=%~d0%~p0

ffmpeg -f image2 -r 60 -i %p%images\image_%%01d.png -c:v libx264 -crf 0 -preset veryslow -c:a libmp3lame -b:a 320k -y ./video.mp4

PAUSE