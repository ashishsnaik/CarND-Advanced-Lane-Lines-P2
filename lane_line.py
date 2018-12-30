from utils import *

# Define a class to receive the characteristics of each line detection
class Line:

    def __init__(self, enable_smoothing=True, nsmooth=5, diff_thresholds=[0.0003, 0.03, 85.0],
                 xm_per_pix=3.7/666, ym_per_pix=27.432/720):

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]
        self.current_fit = []
        # radius of curvature of the line in meters
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # x intercept of the line in the warped image
        self.x_intercept = None

        # enable smoothing?
        self.enable_smoothing = enable_smoothing
        # number of last frames to consider for polynomial smoothing
        self.nsmooth = nsmooth
        # diff thresholds
        self.diff_thresolds = np.array(diff_thresholds, dtype='float')
        # x and y meters per pixel
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix

    def reset(self):
        # reset appropriate member variables
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.current_fit = []
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0,0,0], dtype='float')
        self.allx = None
        self.ally = None
        self.x_intercept = None


    def set_all_xy(self, allx, ally):
        self.allx = allx
        self.ally = ally

    def is_detected(self):
        return self.detected

    def get_best_fit_data(self):
        return self.best_fit, self.bestx

    def get_radius_of_curvature(self):
        return self.radius_of_curvature

    def determine_best_fit(self, current_fit, h=720):

        Y = np.linspace(0, h - 1, h)

        # check whether we have a current fit
        if self.enable_smoothing is True:
            if current_fit is not None:
                self.detected = True
                if self.best_fit is not None:
                    self.diffs = np.abs(current_fit - self.best_fit)

                # if the diffs are greater than the thresholds, smooth the poly-fit
                if len(self.current_fit) > 0 and all(self.diffs > self.diff_thresolds) is True:
                    # smooth over last 'n' + current observations
                    self.best_fit = np.mean(np.append(self.current_fit, [current_fit], axis=0), axis=0)
                else:
                    self.best_fit = current_fit
            else:
                # if no current fit, do nothing, we'll use the previous best fit, if available
                # (just record that a line was not detected this time)
                self.detected = False
                # smooth over last 'n' observations
                if len(self.current_fit) > 0:
                    self.best_fit = np.mean(self.current_fit, axis=0)

            # only the best fit coefs are appended to the list of current fits
            self.current_fit.append(self.best_fit)
            # keep only latest 'nsmooth' fits
            if len(self.current_fit) > self.nsmooth:
                self.current_fit = self.current_fit[len(self.current_fit) - self.nsmooth:]

            # if self.best_fit is not None:
            #     self.bestx = self.best_fit[0] * Y ** 2 + self.best_fit[1] * Y + self.best_fit[2]
            #     self.x_intercept = self.bestx[len(self.bestx)-1]
        else:
            # only record the current fit if it's not None
            if current_fit is not None:
                self.detected = True
                self.best_fit = self.current_fit = current_fit
                # self.bestx = self.recent_xfitted = \
                #     self.best_fit[0] * Y ** 2 + self.best_fit[1] * Y + self.best_fit[2]
            else:  # use whatever the previous best fit is, if one is present
                self.detected = False

        # calculate bestx, x-intercept, and radius of curvature for the lane line
        if self.best_fit is not None:
            self.bestx = self.best_fit[0] * Y ** 2 + self.best_fit[1] * Y + self.best_fit[2]
            self.x_intercept = self.bestx[len(self.bestx) - 1]
            self.calculate_radius_of_curvature(mdist_from_pov=0, h=h)

    def fit_poly(self, binary):

        h, w = binary.shape

        if self.allx is not None and self.ally is not None:
            fit = np.polyfit(self.ally, self.allx, 2)
        else:
            print("WARNING: lane_line::Line - allx or ally OR both are None.")
            fit = None

        return self.determine_best_fit(fit, h)

    def calculate_radius_of_curvature(self, mdist_from_pov=0, h=720):

        if self.bestx is not None:
            mx, my = self.xm_per_pix, self.ym_per_pix
            Y = np.linspace(0, h-1, h)
            X = self.bestx

            lookahead_pix = min(mdist_from_pov/my, h-1)
            y_eval = h-lookahead_pix-1

            m_fit = np.polyfit(Y * my, X * mx, 2)
            A, B, _ = m_fit
            self.radius_of_curvature = ((1 + (2*A*y_eval*my + B)**2)**(3/2))/np.abs(2*A)
        else:
            print("WARNING: bestx in None. Couldn't calculate ROC.")
