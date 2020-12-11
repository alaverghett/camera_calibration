import numpy as np
import matlab.engine
from pprint import pprint

# implements all functionality required for the task
class assignment4():
    def __init__(self):
        self.R1 = None
        self.T1 = None
        self.f1 = None
        self.K1 = None
        self.R2 = None
        self.T2 = None
        self.f2 = None
        self.K2 = None
        self.cx = None
        self.cy = None
        self.X1 = []
        self.X2 = []
    
    # the files testData.m and tsai.m contain the implementation of tsai calibration
    # code taken from here:
    # https://github.com/simonwan1980/Tsai-Camera-Calibration
    def tsaiCalibration(self):
        eng = matlab.engine.start_matlab()
        eng.testData(nargout=0)
        self.cx = eng.workspace['Cx']
        self.cy = eng.workspace['Cy']
        self.R1 = np.asarray(eng.workspace['R1'])
        self.T1 = np.asarray(eng.workspace['T1'])
        self.f1 = np.asarray(eng.workspace['f1'])
        self.K1 = np.asarray([[-self.f1,0,self.cx],[0,-self.f1,self.cy],[0,0,1]])
        self.R2 = np.asarray(eng.workspace['R2'])
        self.T2 = np.asarray(eng.workspace['T2'])
        self.f2 = np.asarray(eng.workspace['f2'])
        self.K2 = np.asarray([[-self.f2,0,self.cx],[0,-self.f2,self.cy],[0,0,1]])
        X1_raw = np.asarray(eng.workspace['X1'])
        X2_raw = np.asarray(eng.workspace['X2'])
        eng.quit()
        
        # convert to homogeneous
        for i in range(0,8):
            self.X1.append([int(X1_raw[i][0]),int(X1_raw[i][1]),1])
            self.X2.append([int(X2_raw[i][0]),int(X2_raw[i][1]),1])
        
        
        
    # given the camera parameters K, R, T and all xi points projected in all i cameras
    # Estimate each Xi 3d point in the world coordinate system
    def triangulate(self, epsilon=0):
        # keep track of the estimates made from each point in each camera
        estimates = {}
        # the intrinsic and extrinsic parameters
        RL = [self.R1,self.R2]
        TL = [self.T1,self.T2]
        KL = [self.K1,self.K2]

        # for each point
        for pt in range(0,8):               
            # define initial estimates
            r = 999
            rPrev = 1000
            p = np.asarray([[0,0,0]])
            I = np.identity(3)

            # iteratively update estimate of 3D locations using nonlinear least squares
            while(r < (rPrev - epsilon)):
                rPrev = r
                dR = 0
                dP1 = 0
                dP2 = 0
                # for each camera
                for c in range(0,2):
                    # calculate cj
                    cj = -RL[c].T @ TL[c] # 3x1
                    # get the normalized unit vector
                    vec = np.asarray([RL[c].T @ np.linalg.inv(KL[c]) @ self.X1[pt]]).T
                    norm = np.linalg.norm(vec)
                    vHatJ = np.divide(vec, norm)
                    # and dj
                    dj = vHatJ.T @ (p - cj)
                    # estimate of the point from jth camera
                    dP1 += I - (vHatJ @ vHatJ.T)
                    dP2 += (I - (vHatJ @ vHatJ.T)) @ cj
                    # residual of jth camera
                    dR += ((vHatJ @ vHatJ.T) @ (p - cj)) - (p - cj)

                # get total residual
                r = np.linalg.norm(dR) ** 2
                # and the total estimate
                p = np.linalg.inv(dP1) @ dP2
                
            # add the estimated 3d point
            estimates[pt] = p
        
        # output results
        print("Predicted coordinates of key points")
        pprint(estimates)
        

    def main(self):
        self.tsaiCalibration()
        self.triangulate()



###################
# BEGIN EXEC HERE #
###################
src = assignment4()
src.main()