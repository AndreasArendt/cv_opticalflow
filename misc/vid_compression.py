import moviepy.editor as mp

clip = mp.VideoFileClip("./data/uav_2.mp4")
#clip_resized = clip.resize(height=360, width=360) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
clip_resized = clip.resize(newsize=(320,320))
clip_resized.write_videofile("./data/uav_2_resized.mp4")

