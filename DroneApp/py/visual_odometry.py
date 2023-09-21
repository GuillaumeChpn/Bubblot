import numpy as np
import cv2
import time

frames = 20

def calculate_angles_between_matrices(matrix1, matrix2): 
	dot_products = np.sum(matrix1 * matrix2, axis=1) 
	norms_product = np.linalg.norm(matrix1, axis=1) * np.linalg.norm(matrix2, axis=1) 
	cos_angles = dot_products / norms_product 
	angles_in_radians = np.arccos(cos_angles) 
	angles_in_degrees = np.degrees(angles_in_radians) 
	return angles_in_degrees

def LiveTracking(cap,lk_params,width,height,old_gray,old_frame,feature_params,nb_pts,color,p0,frame_count,mov1,mov2,mov3,mov4,dusk):
	
	while frame_count<frames+1:
		
		# Capture frame-by-frame
		ret, frame = cap.read()
		
		# if frame is read correctly ret is True
		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			break
		
		frame_gray = cv2.cvtColor(frame,
								cv2.COLOR_BGR2GRAY)

		# calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
											frame_gray,
											p0, None,
											**lk_params)

		
		# Select good points
		if p1 is not None:
			good_new = p1[st == 1]
		if p0 is not None:
			good_old = p0[st == 1]

		###For yaw angle calculation
		yaw_angle=0
		if frame_count==0:
			src_pts=good_old
			
		if ((np.count_nonzero(st==0)>0)&(frame_count>0)):
			for bad in reversed(np.argwhere(st==0)[:,0]):
				if bad<np.shape(src_pts)[0]:
					src_pts=np.delete(src_pts,bad,0)

		diff_pix=good_new-good_old
		

		# Find points in each quadrant
		if (np.shape(good_old)[0]>0):
			idx_topleft=np.argwhere((good_old[:,0]<=width/2) & (good_old[:,1]<=height/2))
			idx_bottomleft=np.argwhere((good_old[:,0]<=width/2) & (good_old[:,1]>height/2))
			idx_topright=np.argwhere((good_old[:,0]>width/2) & (good_old[:,1]<=height/2))
			idx_bottomright=np.argwhere((good_old[:,0]>width/2) & (good_old[:,1]>height/2))

		
		
		if((idx_topleft is not None)&(len(idx_topleft)>0)&(len(idx_bottomleft)>0)&(len(idx_topright)>0)&(len(idx_bottomright)>0)&(np.shape(diff_pix)[0]>np.shape(idx_topleft)[0])):
			
			topleft_diff=np.mean(diff_pix[idx_topleft],axis=0)
			bottomleft_diff=np.mean(diff_pix[idx_bottomleft],axis=0)
			topright_diff=np.mean(diff_pix[idx_topright],axis=0)
			bottomright_diff=np.mean(diff_pix[idx_bottomright],axis=0)
			notvalid=False
		else:
			notvalid=True

		
		lines = np.zeros_like(old_frame)

		#Calculate what the displacement was in the last 10 frames
		if frame_count == frames:
			
			if notvalid:
				type_mov="Not enough points"
				v_corr=[0,0,0]
				t_elapsed=0
				pass
			else:
				dawn=time.time()
				t_elapsed=dawn-dusk
				v_corr=np.zeros((3,1))
				norm1=np.linalg.norm(mov1)
				norm2=np.linalg.norm(mov2)
				norm3=np.linalg.norm(mov3)
				norm4=np.linalg.norm(mov4)
				main_vec=np.argmax([norm1,norm2,norm3,norm4])

				mov1_norm=(mov1/norm1)
				mov2_norm=(mov2/norm2)
				mov3_norm=(mov3/norm3)
				mov4_norm=(mov4/norm4)

				main_vec_norm=[mov1_norm,mov2_norm,mov3_norm,mov4_norm][main_vec]
				

				# # # #Draw arrows to visualize displacement
				# # # cv2.arrowedLine(frame, (int(width/4), int(height/4)), (int(width/4 +10*(mov1[0][0])), int(height/4 +10*(mov1[0][1]))), (0,255,0), 5, tipLength=0.3)
				# # # cv2.arrowedLine(frame, (int(width/4), int(3*height/4)), (int(width/4 +10*(mov2[0][0])), int(3*height/4 +10*(mov2[0][1]))), (0,255,0), 5, tipLength=0.3)
				# # # cv2.arrowedLine(frame, (int(3*width/4), int(height/4)), (int(3*width/4 +10*(mov3[0][0])), int(height/4 +10*(mov3[0][1]))), (0,255,0), 5, tipLength=0.3)
				# # # cv2.arrowedLine(frame, (int(3*width/4), int(3*height/4)), (int(3*width/4 +10*(mov4[0][0])), int(3*height/4 +10*(mov4[0][1]))), (0,255,0), 5, tipLength=0.3)
				# # # cv2.arrowedLine(frame, (int(width/2), int(height/2)), (int(width/2 +(mov1[0][0]+mov2[0][0]+mov3[0][0]+mov4[0][0])), int(height/2 +(mov1[0][1]+mov2[0][1]+mov3[0][1]+mov4[0][1]))), (255,0,0), 5,tipLength=0.3)	


				######### ESSENTIAL MATRIX METHOD (NEEDS CAMERA MATRIX) ##########
				dst_pts=good_new[:np.shape(src_pts)[0],:]
				
				

				# # # E,mask=cv2.findEssentialMat((src_pts),(dst_pts),cameraMatrix=Kcam, method=cv2.RANSAC)
				# # # dist_pts=np.linalg.norm(dst_pts-src_pts,axis=1)

				# # # retval, Rot, t, mask = cv2.recoverPose(E,src_pts,dst_pts,cameraMatrix=Kcam,mask=mask)
				# # # src_pts=good_new
				
				
				# # # diag_R=np.diag(Rot)
				# # # diag_norm=np.abs(diag_R)

				# # # if retval==0:
				# # # 	print("Rotation matrix not accurate enough")
				##### ESSENTIAL MATRIX DECOMPOSITION ######
				# # # else:
				# # # 	if(np.amax([norm1,norm2,norm3,norm4])<5):
				# # # 		print("Displacement too small, nothing to do...")
				# # # 	if (np.amin(np.abs(diag_R))>0.999):
				# # # 		# print("Pure Zoom or translate")
				# # # 		if ((np.abs(t[2]/t[0])>2) & (np.abs(t[2]/t[1])>2)):
				# # # 			if(t[2]<0):
				# # # 				print("zoom in")
				# # # 			else:
				# # # 				print("zoom out")
				# # # 		else:
				# # # 			print("Translation")
				# # # 	else:
				# # # 		if ((diag_norm[2]<diag_norm[0])&(diag_norm[2]<diag_norm[1])):
				# # # 			if ((0.8*diag_norm[1]>diag_norm[2])|(1.25*diag_norm[1]<diag_norm[2])):
				# # # 				print("Yaw rotation")
				# # # 			else:
				# # # 				print("Tilting or Rolling")
				# # # 		elif(diag_norm[2]<diag_norm[0]):
				# # # 			print("Tilting")
				# # # 		elif(diag_norm[2]<diag_norm[1]):
				# # # 			print("Rolling")

						# print("rotate")

				
				#Dot product between the two diagonals
				m14=np.dot(mov1_norm[0],mov4_norm[0])
				m23=np.dot(mov2_norm[0],mov3_norm[0])

				#Displacement of the four main vectors
				movs=[np.linalg.norm(mov1),np.linalg.norm(mov2),np.linalg.norm(mov3),np.linalg.norm(mov4)]
				v_tot=[mov1[0][0]+mov2[0][0]+mov3[0][0]+mov4[0][0],mov1[0][1]+mov2[0][1]+mov3[0][1]+mov4[0][1]]
				v_tot_norm=v_tot/np.linalg.norm(v_tot)

				#Conditions to assess a direction to each 10-frames pack : shifting (translation), Rotation (yaw) and zooming :
				
				if(np.linalg.norm(v_tot)<20):
					type_mov="no movement"
					v_corr=[0,0,0]
				else:
					if((m14>0.9)&(m23>0.9)):
						ratio12=np.linalg.norm(mov1)/np.linalg.norm(mov2)
						ratio14=np.linalg.norm(mov1)/np.linalg.norm(mov4)
						yaw_angle=0
						#If all vectors magnitudes are similar, the translation is planar
						if (((ratio12>0.8) & (ratio12<1.25)) &	((ratio14>0.8) & (ratio14<1.25))):
							if (np.linalg.norm([v_tot[0],v_tot[1]])>50):
								type_mov="shift"
								v_corr=[v_tot_norm[0],v_tot_norm[1],0]
							else:
								type_mov="no movement"
								v_corr=[0,0,0]
						#Else a zoom is added to the translation
						else:
							type_mov="shift zoom"
							v_corr=[v_tot_norm[0],v_tot_norm[1],5*np.sin(np.amax((movs)-np.amin(movs))*np.pi/200)]

					else:

						m14_diag=np.cross(np.sign(mov1_norm[0]),np.sign(mov4_norm[0]))
						m23_diag=np.cross(np.sign(mov2_norm[0]),np.sign(mov3_norm[0]))
						
						#Find the directions of the quadrant vectors
						c1=mov1_norm[0][0]+mov1_norm[0][1]
						c2=mov2_norm[0][0]-mov2_norm[0][1]
						c3=mov3_norm[0][0]-mov3_norm[0][1]
						c4=mov4_norm[0][0]+mov4_norm[0][1]

						tot_ext=np.count_nonzero([(c1<0) , (c2<0) , (c3>0) ,(c4>0)])

						sign_x=np.sign(mov1_norm[0][0])+np.sign(mov2_norm[0][0])+np.sign(mov3_norm[0][0])+np.sign(mov4_norm[0][0])
						sign_y=np.sign(mov1_norm[0][1])+np.sign(mov2_norm[0][1])+np.sign(mov3_norm[0][1])+np.sign(mov4_norm[0][1])

						h1=main_vec_norm[0][0]+main_vec_norm[0][1]
						h2=main_vec_norm[0][0]-main_vec_norm[0][1]

						#Condition for the displacement to be a rotation (very similar to zooming), trust the process:
						if ((((m14_diag)*(m23_diag)>0) & (sign_x !=0)) | (((m14_diag)*(m23_diag)<0)&(np.abs(sign_x)!=4)) | (((m14_diag)*(m23_diag)==0)&(((tot_ext<4)&(tot_ext>0))|((tot_ext==0) & ((np.abs(mov1_norm[0][0]/mov1_norm[0][1])<1/3) |(np.abs(mov1_norm[0][0]/mov1_norm[0][1])>3)))))):
							yaw_angle=np.mean(calculate_angles_between_matrices(src_pts,dst_pts))
							if (((main_vec==0) & (h2<0))|((main_vec==3) & (h2>0))|((main_vec==1) & (h1>0))|((main_vec==2) & (h1<0))):
								type_mov="Yaw left"
								# v_corr=[0,0,-1]
								v_corr=[0,0,0]
								
							else:
								type_mov="Yaw right"
								# v_corr=[0,0,1]
								v_corr=[0,0,0]
								yaw_angle=-yaw_angle
								
						else:

							yaw_angle=0												
							if (((main_vec==0) & (h1>0))|((main_vec==3) & (h1<0))|((main_vec==1) & (h2>0))|((main_vec==2) & (h2<0))):
								type_mov="zoom out"
								# v_corr=[-v_tot_norm[0],-v_tot_norm[1],-1]
								v_corr=[0,0,-2]
							else:
								type_mov="zoom in"
								# v_corr=[-v_tot_norm[0],-v_tot_norm[1],1]
								v_corr=[0,0,1]
					
					# print("Correcting vector: ",v_corr)
					
			mov1=0
			mov2=0
			mov3=0
			mov4=0

					
					# print("time",t_elapsed)
				
		else:
			if notvalid:
				pass
			else:
				mov1+=topleft_diff
				mov2+=bottomleft_diff
				mov3+=topright_diff
				mov4+=bottomright_diff
		

		mask = np.zeros_like(old_frame)
		# draw the tracks

		# # # # for i, (new, old) in enumerate(zip(good_new,good_old)):
		# # # # 	a, b = new.ravel()
		# # # # 	c, d = old.ravel()
		# # # # 	if i < len(color):
		# # # # 		mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
		# # # # 		frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
		# # # # 	else:
		# # # # 		# Generate new color for newly added corners
		# # # # 		new_color = np.random.randint(0, 255, (1, 3))
		# # # # 		color = np.vstack((color, new_color))
		# # # # 		mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), new_color[0].tolist(), 2)
		# # # # 		frame = cv2.circle(frame, (int(a), int(b)), 5, new_color[0].tolist(), -1)
		# img = cv2.add(frame, lines)
		img=frame
		
		
		# # # cv2.imshow('frame', img)
		
		# # # k = cv2.waitKey(1)
		# # # if k == 27:
		# # # 	break
		# Updating Previous frame and points
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1, 1, 2)
		if frame_count % 2 == 0:
			# Add new corners every 3 frames
			new_corners = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
			if ((new_corners is not None) & (np.shape(p0)[0]>1)):
				# Check distance to existing corners
				for corner in new_corners:
					dist = np.linalg.norm(p0.squeeze() - corner.squeeze(), axis=1)
					if np.min(dist) > 10:  # Minimum distance threshold
						p0 = np.concatenate((p0, corner.reshape(-1, 1, 2)), axis=0)
					
						if len(color) < nb_pts:
							new_color = np.random.randint(0, 255, (1, 3))
							color = np.vstack((color, new_color))
							break

		frame_count += 1
		# print("frame ",frame_count)
	return(type_mov,v_corr,t_elapsed,yaw_angle,p0,old_gray)
		

def UnderCamera():
	# Choose video
	
	cap = cv2.VideoCapture(1)

	if not cap.isOpened():
		
		print("Cannot open camera")
		exit()

	# nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	notvalid=False

	#Load intrinsic Camera Parameters matrix
	# # Kcam=np.load("K.npy")

	# params for corner detection
	nb_pts=200
	feature_params = dict( maxCorners = nb_pts,
						qualityLevel = 0.3,
						minDistance = 10,
						blockSize = 10 )
	
	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize = (15, 15),
					maxLevel = 3,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
								30, 0.03))

	# Create some random colors
	color = np.random.randint(0, 255, (nb_pts, 3))
	
	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	# print(ret,flush=True)

	old_gray = cv2.cvtColor(old_frame,
							cv2.COLOR_BGR2GRAY)
	
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
								**feature_params)
	height=np.shape(old_frame)[0]
	width=np.shape(old_frame)[1]

	# Create a mask image for drawing purposes
	mask = np.zeros_like(old_frame)
	# # # cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	frame_count = 0  # Counter to track frame number

	#Initialization of the four quadrants displacements
	mov1=0
	mov2=0
	mov3=0
	mov4=0

	
	#Travel the frames
	while True:
		
		dusk= time.time()
		type_nov,vec_corr,t_elapsed,yaw_angle,pts0,old_g=LiveTracking(cap,lk_params,width,height,old_gray,old_frame,feature_params,nb_pts,color,p0,frame_count,mov1,mov2,mov3,mov4,dusk)
		p0=pts0
		old_gray=old_g
		print(type_nov,vec_corr,t_elapsed,yaw_angle,flush=True,sep="\n",end="")
	# cv2.destroyAllWindows()
	# cap.release()

	# return 0
UnderCamera()
