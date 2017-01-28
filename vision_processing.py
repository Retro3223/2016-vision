import math
import pygrip
import cv2
import numpy
from networktables import NetworkTable
from target import (
    Target
)
from numpy_pool import (
    NumpyPool
)
from utils import (
    mm_to_in,
    into_uint8,
    into_uint16_mask,
    munge_floats_to_img,
    flatten_contours,
    rgbhex2bgr,
    minAreaBox,
)
from angles import (
    h_angle,
    v_angle
)
from trajectory import (
    on_trajectory
)
from data_logger import DataLogger, Replayer
from xyz_converter import (
    depth_to_xyz2,
    distance,
    midpoint,
)
try:
    import libpclproc
except:
    libpclproc = None

# display modes
DISP_DEPTH = 0
DISP_RAW_IR = 1
DISP_IR_MASK1 = 2
DISP_IR_MASK2 = 3
DISP_ALL_CONTOURS = 4
DISP_KEPT_CONTOURS = 5
DISP_IR_MASK3 = 6
DISP_EDGES = 7


class Vision:
    def __init__(self, shape=(240, 320), use_sensor=True):
        pool = self.pool = NumpyPool(shape=shape)
        shape3 = (shape[0], shape[1], 3)
        self.depth = pool.get_raw()
        self.ir = pool.get_raw()
        self.display = pool.get_color()
        self.mask8 = pool.get_gray()
        self.unblurred_mask8 = pool.get_gray()
        self.mask16 = pool.get_raw()
        self.interesting_depths = numpy.zeros(shape=shape, dtype='uint16')
        self.target_depths = pool.get_raw()
        self.contour_img = pool.get_color()
        self.xyz = pool.get_xyz()

        self.is_hg_position = True
        self.hg_angle = 35.0 # degrees
        self.mode = 0
        self.area_threshold = 10
        self.rat_min = 0.1
        self.rat_max = 0.6
        self.perimeter_threshold = 10
        self.n_shiniest = 1
        self.contours = []
        self.targets = []
        self.max_target_width = 70
        self.max_target_height = 70
        self.max_target_count = 3
        self.center_x = 160
        self.center_y = 120
        self.angle = 45
        self.exit_velocity = 6200 # mm/s
        self.targeting = False
        self.avg_mm = -1
        self.min_dist = 1000
        self.sd = NetworkTable.getTable("SmartDashboard")
        self.use_sensor = use_sensor

    def __enter__(self):
        if self.use_sensor:
            import structure3223
            structure3223.init()
        return self

    def __exit__(self, *args):
        if self.use_sensor:
            import structure3223
            structure3223.destroy()

    def set_display(self):
        if self.mode == 0:
            temp = self.pool.get_gray()
            into_uint8(self.depth, dst=temp)
            cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR, dst=self.display)
            self.pool.release_gray(temp)
        elif self.mode == 1:
            temp = self.pool.get_gray()
            into_uint8(self.ir, dst=temp)
            cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR, dst=self.display)
            self.pool.release_gray(temp)
        elif self.mode == 2:
            cv2.cvtColor(self.mask8, cv2.COLOR_GRAY2BGR, dst=self.display)
        elif self.mode == 3:
            temp = self.pool.get_gray()
            into_uint8(self.interesting_depths, dst=temp)
            cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR, dst=self.display)
            self.pool.release_gray(temp)
        elif self.mode == 4:
            munge_floats_to_img(self.xyz, dst=self.display)
        else:
            numpy.copyto(dst=self.display, src=self.contour_img)

    def setup_mode_listener(self):
        self.sd.addTableListener(self.value_changed)

    def value_changed(self, table, key, value, is_new):
        if key == "structureMode":
            if value in [0, 1, 2, 3, 4, 5]:
                self.set_mode(value)
        elif key == "shooter_pitch2":
            self.angle = value
        elif key == "exit_velocity":
            self.exit_velocity = value

    def get_depths(self):
        import structure3223
        structure3223.read_frame(depth=self.depth, ir=self.ir)
        self.flip_inputs()

    def get_recorded_depths(self, replayer, i):
        results = replayer.load_frame(i)
        self.depth, self.ir = results['depth'], results['ir']
        if 'xyz' in results:
            self.xyz = results['xyz']

    def flip_inputs(self):
        cv2.flip(self.depth, 1, dst=self.depth)
        cv2.flip(self.ir, 1, dst=self.ir)

    def display_depth(self):
        if self.mode == DISP_DEPTH:
            temp = self.pool.get_gray()
            into_uint8(self.depth, dst=temp)
            cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR, dst=self.display)
            self.pool.release_gray(temp)

    def display_raw_ir(self):
        if self.mode == DISP_RAW_IR:
            temp = self.pool.get_gray()
            into_uint8(self.ir, dst=temp)
            cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR, dst=self.display)
            self.pool.release_gray(temp)

    def display_ir_mask(self, mask):
        if self.mode == DISP_IR_MASK1:
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR, dst=self.display)

    def display_ir_mask2(self, mask):
        if self.mode == DISP_IR_MASK2:
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR, dst=self.display)
            """
            temp8 = self.pool.get_gray()
            into_uint8(mask, dst=temp8)
            cv2.cvtColor(temp8, cv2.COLOR_GRAY2BGR, dst=self.display)
            self.pool.release_gray(temp8)
            """

    def display_ir_mask3(self):
        if self.mode == DISP_IR_MASK3:
            cv2.cvtColor(self.mask8, cv2.COLOR_GRAY2BGR, dst=self.display)
            if len(self.contours) == 0: return
            rect = cv2.boundingRect(self.contours[0])
            cv2.rectangle(self.display, 
                (rect[0], rect[1]), 
                (rect[0]+rect[2], rect[1]+rect[3]), 
                (0, 0, 255), 1)
            center_i = rect[0] + rect[2] // 2
            center_j = rect[1] + rect[3] // 2
            print ('center: ', center_i, center_j)
            print (' c x: ', 160 - center_i)
            print (' c x:: ', self.xyz[0, center_j, center_i])
            return self.display

    def publish_xoffset(self):
        if len(self.contours) == 0: 
            self.sd.putBoolean("seesHighGoal", False)
            return
        rect = cv2.boundingRect(self.contours[0])
        center_i = rect[0] + rect[2] // 2
        center_j = rect[1] + rect[3] // 2
        x = self.xyz[0, center_j, center_i]
        x_pixel_offset = 160 - center_i
        self.sd.putBoolean("seesHighGoal", True)
        self.sd.putNumber("xOffsetHighGoal", x_pixel_offset)
    
    def hg_draw_hud(self):
        if self.mode == 2:
           z0 = 2000
           """self.display[:] = 0"""
           cv2.circle(self.display,(320//2,240//2),2,(0,255,0),1)
           #cv2.circle(self.display,(100,60),8,(0,0,255),-1)
           if len(self.contours) ==0: return
           contour = self.contours[0]
           a = cv2.boundingRect(contour) 
           #print(a)
           b = ((a[1]+(a[1]+a[3]))//2,(a[0]+(a[0]+a[2]))//2)
           #print(b)
           x_goal = self.xyz[0,b[1],b[0]]
           z_goal = self.xyz[2,b[1],b[0]]
           dz = z_goal - z0
           cv2.circle(self.display,b,8,(0,0,255),-1)

    def process_depths(self):
        """
        """
        self.display_depth()
        self.display_raw_ir()

        if self.is_hg_position:
            if self.mode == DISP_IR_MASK3:
                self.display[:] = 0
            self.hg_mask_shiny()
            depth_to_xyz2(depth=self.depth, xyz=self.xyz)
            self.hg_filter_shiniest()
            self.hg_find_edges()
            self.hg_draw_hud()
            self.publish_xoffset()
        else:
            pass
        """
        self.zero_out_min_dists()
        self.mask_shiny()
        self.filter_shiniest()
        cv2.bitwise_and(self.depth, self.mask16, dst=self.interesting_depths)
        depth_ixs = numpy.nonzero(self.interesting_depths)
        depth_to_xyz2(depth=self.depth, xyz=self.xyz)
        count = len(depth_ixs[0])
        if count != 0:
            sum = numpy.sum(self.interesting_depths[depth_ixs])
            self.avg_mm = sum / count
        else:
            avg_mm = -1
        rects = [(cv2.boundingRect(c), c) for c in self.contours]
        self.targets = self.build_targets(rects)

        chosen_j = self.choose_target()
        target_mask = self.pool.get_raw()

        for j, target in enumerate(self.targets):
            if j == chosen_j:
                target.draw_color = (0, 255, 0)
            target.draw(self.contour_img)
            target_mask[:] = 0
            cv2.drawContours(
                target_mask,
                target.contours, -1, (0xffff), cv2.FILLED)
            cv2.bitwise_and(
                self.depth, target_mask, dst=self.target_depths)
            depth_ixs = numpy.nonzero(self.target_depths)
            count = len(depth_ixs[0])
            if count != 0:
                sum = numpy.sum(self.target_depths[depth_ixs])
                target.avg_mm = sum / count
                target.theta = h_angle(target.center_x, CX=self.center_x)
                target.theta_v = v_angle(target.center_y, CY=self.center_y)

            if j == chosen_j:
                self.sd.putNumber("target_dist", target.avg_mm)
                self.sd.putNumber("target_theta", target.theta)
                self.sd.putNumber("target_theta_v", target.theta_v)
                cv2.putText(self.contour_img,
                    ("d= %.2f in" % target.avg_in), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.putText(self.contour_img,
                    ("theta= %.2f" % target.theta), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.putText(self.contour_img,
                    ("thetav= %.2f" % target.theta_v), (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
            else:
                self.sd.putNumber("target_dist", -1)
                self.sd.putNumber("target_theta", 1000)
                self.sd.putNumber("target_theta_v", 1000)


        self.pool.release_raw(target_mask)
        self.predict_impact(chosen_j)
        self.measure_target(chosen_j)
        #self.set_display()
        """

    def zero_out_min_dists(self):
        ixs = self.depth < self.min_dist
        self.depth[ixs] = 0
        self.ir[ixs] = 0

    def hg_mask_shiny(self):
        ir_temp = self.pool.get_raw()
        numpy.copyto(ir_temp, self.ir)
        ir_temp[:,:] = 0
        # threshold raw ir data
        ixs = self.ir > 200
        ir_temp[ixs] = 0xffff
        # ignore shiny things that are too close
        ixs = self.depth < 1400 # mm
        ixs &= self.depth != 0
        ir_temp[ixs] = 0
        # and too far away
        ixs = self.depth > 9000 # mm
        ir_temp[ixs] = 0
        into_uint8(ir_temp, dst=self.unblurred_mask8)
        self.display_ir_mask(self.unblurred_mask8)
        # blur the shiny, reduce the noise for contour finding
        pygrip.blur(
            self.unblurred_mask8, 
            pygrip.MEDIAN_BLUR, 
            radius=1, dst=self.mask8)

        ixs = self.mask8 > 80 
        self.mask8[ixs] = 255
        ixs = self.mask8 < 80 
        self.mask8[ixs] = 0
        self.display_ir_mask2(self.mask8)
        # grr threshold operates on matrices of unsigned bytes
        into_uint16_mask(self.mask8, dst=self.mask16)

        self.pool.release_raw(ir_temp)

    def set_mode(self, mode_num):
        self.mode = mode_num

    def display_all_contours(self, all_contours):
        if self.mode == DISP_ALL_CONTOURS:
            # show all contours in blue
            # show kept contours in green
            temp_depth = self.pool.get_gray()
            temp_color = self.pool.get_color()
            #into_uint8(self.depth, dst=temp_depth)
            #cv2.cvtColor(temp_depth, cv2.COLOR_GRAY2BGR, dst=temp_color)
            temp_color[:] = 255
            cv2.drawContours(
                temp_color, all_contours, -1, (255, 0, 0), 1)
            numpy.copyto(dst=self.display, src=temp_color)
            self.pool.release_gray(temp_depth)
            self.pool.release_color(temp_color)

    def display_kept_contours(self, kept_contours):
        if self.mode == DISP_KEPT_CONTOURS:
            # show all contours in blue
            # show kept contours in green
            temp_depth = self.pool.get_gray()
            temp_color = self.pool.get_color()
            into_uint8(self.depth, dst=temp_depth)
            cv2.cvtColor(temp_depth, cv2.COLOR_GRAY2BGR, dst=temp_color)
            cv2.drawContours(
                temp_color, kept_contours, -1, (0, 255, 0), 1)
            numpy.copyto(dst=self.display, src=temp_color)
            self.pool.release_gray(temp_depth)
            self.pool.release_color(temp_color)

    def hg_filter_shiniest(self):
        # find contours of shiny things
        # grr, findContours modifies its input image
        contour_mask = self.pool.get_gray()
        numpy.copyto(contour_mask, self.mask8)
        things = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_contours = things[1]
        self.display_all_contours(all_contours)
        contours = [c for c in all_contours if self.hg_filter_contours(c)]
        contours.sort(key=lambda c: -cv2.contourArea(c))
        self.display_kept_contours(contours)
        self.mask16[:] = 0
        self.mask8[:] = 0
        cv2.drawContours(self.mask8, contours, -1, (0xff), cv2.FILLED)
        cv2.drawContours(self.mask16, contours, -1, (0xffff), cv2.FILLED)
        display = self.display_ir_mask3()
        self.contours = contours
        self.pool.release_gray(contour_mask)

    def hg_filter_contours(self, contour):
        """
        is this a contour that is probably of the high goal?
        find the real width and height of the contour 
        (not adjusted for rotation)
        is it too wide? -> no
        is it too tall? -> no
        is it too small? -> no
        """
        mask = self.pool.get_gray()
        area = cv2.contourArea(contour)
        if area < 10:
            # smaller than 10 pixels? not actionable
            return False
        # isolate the mask enclosed by this contour
        (x, y, w, h) = cv2.boundingRect(contour)
        mask_part = mask[y:y+h, x:x+w]
        mask_part[:] = 0
        cv2.drawContours(mask, [contour], -1, (255,), cv2.FILLED)
        mask8_part = self.unblurred_mask8[y:y+h, x:x+w]
        mask_part &= mask8_part
        # ignore pixels with outlier distances 
        depth_part = self.depth[y:y+h, x:x+w]
        mid_depth = numpy.median(depth_part)
        mask_part[depth_part > mid_depth+400] = 0
        mask_part[depth_part < mid_depth-400] = 0
        # get xyz coords enclosed by this contour
        xyz_part = self.xyz[:, y:y+h, x:x+w]
        ixs = mask_part == 255
        #print (' sh: ', xyz_part.shape)
        x_part = xyz_part[0,:,:][ixs]
        #print ("sh: ", x_part.shape)
        if len(x_part) != 0:
            # vision target is 381 mm x 101 mm or smaller
            # so diagonal is 394 mm or smaller
            # multiply by 1.12 for safety margin,
            # so object must be less than 426 mm across
            # .. and then experimental data suggests 600 is a better max
            # and something that isn't 2 in wide probably isn't the target.
            # probably.
            width = abs(x_part.max() - x_part.min())
            #print (' w: ', width, x_part.max(), x_part.min())
            if width > 600 or width < 50:
                return False

        y_part = xyz_part[1, :, :][ixs]
        if len(y_part) != 0:
            # vision target is 381 mm x 101 mm or smaller
            # we probably won't get it rotated more than 20 degrees
            # and it should be at least an inch tall
            height = abs(y_part.max() - y_part.min())
            max_height = hg_max_apparent_height(20)
            if height > max_height or height < 25:
                return False

        self.pool.release_gray(mask)
        return True

    def hg_median_dist(self, contour):
        mask = self.pool.get_gray()
        (x, y, w, h) = cv2.boundingRect(contour)
        mask_part = mask[y:y+h, x:x+w]
        mask_part[:] = 0
        cv2.drawContours(mask, [contour], -1, (255,), cv2.FILLED)
        mask8_part = self.unblurred_mask8[y:y+h, x:x+w]
        mask_part &= mask8_part
        # ignore pixels with outlier distances 
        depth_part = self.depth[y:y+h, x:x+w]
        mid_depth = numpy.median(depth_part)
        self.pool.release_gray(mask)
        return mid_depth

    def hg_find_edges(self):
        self.detect_edges()

    def display_edges(self, edges):
        if self.mode == DISP_EDGES:
            temp = self.pool.get_gray()
            into_uint8(self.depth, dst=temp)
            cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR, dst=self.display)
            self.pool.release_gray(temp)
            for contour in self.contours:
                (x,y,w,h) = cv2.boundingRect(contour)
                mid_depth = self.hg_median_dist(contour)
                ixs = self.depth <= mid_depth + 200
                ixs &= self.depth >= mid_depth - 200
                self.display[ixs,0] = 97
                self.display[ixs,1] = 206
                self.display[ixs,2] = 202
            for contour in self.contours:
                (x,y,w,h) = cv2.boundingRect(contour)
                box = minAreaBox(contour)
                pt1 = tuple(box[0])
                pt2 = tuple(box[1])
                pt3 = tuple(box[2])
                pt4 = tuple(box[3])
                cv2.line(self.display, pt1, pt2, (0, 0, 255), 1)
                cv2.line(self.display, pt2, pt3, (0, 0, 255), 1)
                cv2.line(self.display, pt3, pt4, (0, 0, 255), 1)
                cv2.line(self.display, pt1, pt4, (0, 0, 255), 1)
                def x(pt):
                    return pt[0]
                redge1 = redge2 = None
                ledge1 = ledge2 = None
                if abs(x(pt1) - x(pt2)) < abs(x(pt2) - x(pt3)):
                    cv2.circle(self.display, pt2, 2, (255, 255, 0), -1)
                    cv2.circle(self.display, pt3, 2, (255, 255, 0), -1)

                    redge1 = self.hg_find_right_edge(pt2, pt3)
                    ledge1 = self.hg_find_left_edge(pt2, pt3)

                    cv2.circle(self.display, pt1, 2, (0, 255, 255), -1)
                    cv2.circle(self.display, pt4, 2, (0, 255, 255), -1)

                    redge2 = self.hg_find_right_edge(pt1, pt4)
                    ledge2 = self.hg_find_left_edge(pt1, pt4)
                else:
                    cv2.circle(self.display, pt1, 2, (255, 255, 0), -1)
                    cv2.circle(self.display, pt2, 2, (255, 255, 0), -1)
                    redge1 = self.hg_find_right_edge(pt1, pt2)
                    ledge1 = self.hg_find_left_edge(pt1, pt2)
                    cv2.circle(self.display, pt3, 2, (0, 255, 255), -1)
                    cv2.circle(self.display, pt4, 2, (0, 255, 255), -1)
                    redge2 = self.hg_find_right_edge(pt3, pt4)
                    ledge2 = self.hg_find_left_edge(pt3, pt4)
                if redge1 != None:
                    cv2.circle(self.display, redge1, 2, (255, 0, 255), -1)
                if redge2 != None:
                    cv2.circle(self.display, redge2, 2, (255, 0, 255), -1)
                if ledge1 != None:
                    cv2.circle(self.display, ledge1, 2, (255, 0, 255), -1)
                if ledge2 != None:
                    cv2.circle(self.display, ledge2, 2, (255, 0, 255), -1)

    def detect_edges(self):
        depth_a = self.depth[:, 0:-1]
        depth_b = self.depth[:, 1:]
        depth_diff = numpy.absolute((depth_a - depth_b).astype('int16'))
        idx = depth_diff < self.area_threshold * 10
        depth_diff[idx] = 0
        depth8 = self.pool.get_gray()
        depth8[:, :] = 255
        depth8[idx] = 0
        #into_uint8(depth_diff, dst=depth8[:, 0:-1])
        """
        things = cv2.findContours(
            depth8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = things[1]
        """
        self.display_edges(depth8)
        #cv2.drawContours(self.display, contours, -1, (244, 66, 241), cv2.FILLED)
        self.pool.release_gray(depth8)

    def hg_find_right_edge(self, pt1, pt2):
        """
        given horizontal segment of bounding box hg vision target,
        find the edge of the pipe by walking that line until you see
        a large depth delta 
        pt1, pt2: (i, j), where i in range (0, 320), j in range (0, 240)
        returns (i, j) of right edge along line segment.
         or None, if none was found
        """
        if pt1[0] > pt2[0]:
            temp = pt1
            pt1 = pt2
            pt2 = temp

        start_pt = ((pt1[0]+pt2[0])//2, pt2[1])
        last_depth = int(self.depth[start_pt[1], start_pt[0]])
        for i in range(start_pt[0], min(pt2[0]+100, 320)):
            depth = int(self.depth[pt2[1], i])
            if depth == 0:
                # hoping we won't see shadow on the right edge, so
                # let's assume any shadow comes from something in
                # front of target
                return None
            ddepth =  depth - last_depth
            last_depth = depth
            if ddepth < -100:
                # we are not at edge, something is in front of target
                # observed negative deltas of down to -46 in valid cases
                return None
            if ddepth > 200:
                # we are at edge
                return (i, pt2[1])
        # .. and I just realized I wrote this assuming horizontal line segments,
        # when that isn't necessarily the case. meh, works okay anyways

    def hg_find_left_edge(self, pt1, pt2):
        """
        given horizontal segment of bounding box hg vision target,
        find the edge of the pipe by walking that line until you see
        a large depth delta 
        pt1, pt2: (i, j), where i in range (0, 320), j in range (0, 240)
        returns (i, j) of left edge along line segment.
         or None, if none was found
        """
        if pt1[0] > pt2[0]:
            temp = pt1
            pt1 = pt2
            pt2 = temp

        start_pt = ((pt1[0]+pt2[0])//2, pt1[1])
        last_depth = int(self.depth[start_pt[1], start_pt[0]])
        for i in range(start_pt[0], max(pt1[0]-100, 0), -1):
            depth = int(self.depth[pt1[1], i])
            if depth == 0:
                # left edges seems to be shadowy, so lets assume
                # shadow actually is edge of target
                return (i, pt1[1])
            ddepth =  depth - last_depth
            last_depth = depth
            if ddepth < -100:
                # we are not at edge, something is in front of target
                # observed negative deltas of down to -46 in valid cases
                return None
            if ddepth > 200:
                # we are at edge
                return (i, pt1[1])

    def measure_target(self, chosen_j):
        from scipy.spatial.distance import pdist
        from scipy.interpolate import interp1d
        if chosen_j is None:
            return
        target = self.targets[chosen_j]
        flattened_contours = flatten_contours(target.contours)
        color = rgbhex2bgr(0xf9c308)
        box = minAreaBox(flattened_contours)
        pt1 = midpoint(box[0], box[1])
        pt2 = midpoint(box[2], box[3])

        things1 = self.measure_target_width_on_segment(pt1, pt2)
        if things1 is not None:
            self.display_measurement(things1)

        pt3 = midpoint(box[0], box[3])
        pt4 = midpoint(box[1], box[2])

        things2 = self.measure_target_width_on_segment(pt3, pt4)
        if things2 is not None:
            self.display_measurement(things2)

    def measure_target_width_on_segment(self, pt1, pt2):
        """
        Given the line segment L defined by 2d points pt1 and pt2 from a camera 
        frame, find the points pt3 and pt4 the nearest points to pt1 and pt2 
        on L that are masked according to self.mask8. Then calculate the 
        distance D between 3d points pt5 and pt6 in self.xyz which 
        correspond to pt3 and pt4.
        return pt3, pt4, D, fx, fy,
            where 
                pt3 = (x, y)
                pt4 = (x, y)
                fx is the function f(distance from pt3 on L) = x
                fy is the function f(distance from pt3 on L) = y
        If anything goes wrong, return None
        """
        from scipy.interpolate import interp1d

        dist2d = distance(pt1, pt2)
        interpx = interp1d([0, dist2d], [pt1[0], pt2[0]])
        interpy = interp1d([0, dist2d], [pt1[1], pt2[1]])
        t = numpy.linspace(0, int(dist2d), int(dist2d)+1)
        xs = numpy.int0(interpx(t))
        ys = numpy.int0(interpy(t))
        ixs, = self.mask8[ys, xs].nonzero()
        if len(ixs) >= 2:
            x1 = xs[ixs[0]]
            y1 = ys[ixs[0]]
            x2 = xs[ixs[-1]]
            y2 = ys[ixs[-1]]
            xyz1 = self.xyz[:, y1, x1]
            xyz2 = self.xyz[:, y2, x2]
            dist3d = distance(xyz1, xyz2)
            interpx2 = lambda d: (x2-x1)*d/dist2d + x1
            interpy2 = lambda d: (y2-y1)*d/dist2d + y1
            return (x1, y1), (x2, y2), dist3d, interpx2, interpy2

    def display_measurement(self, stuff):
        pt1, pt2, dist, fx, fy = stuff
        dist2d = distance(pt1, pt2)
        txt_x = int(fx(dist2d+20))
        txt_y = int(fy(dist2d+20))
        cv2.circle(self.contour_img, pt1, 2, rgbhex2bgr(0xf7ff1e), 
                thickness=2)
        cv2.circle(self.contour_img, pt2, 2, rgbhex2bgr(0xf7ff1e), 
                thickness=2)
        cv2.putText(self.contour_img,
            ("%.2f in" % mm_to_in(dist)), (txt_x, txt_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgbhex2bgr(0xf7ff1e))

    def predict_impact(self, chosen_j):
        def get_angle(y_pixel):
            return self.angle + v_angle(y_pixel, CY=self.center_y)

        def calc_point(d, y_pixel):
            mth = get_angle(y_pixel)
            x = d * math.cos(math.radians(mth))
            y = d * math.sin(math.radians(mth))
            return x, y
        
        def plot_impact(i, d):
            half_angle = math.degrees(math.atan(127. / d))
            pixel_radius = (half_angle / (58. / 320.))
            cv2.circle(self.contour_img, 
                    (self.center_x, i), int(pixel_radius), 
                    (0, 0, 255), thickness=1)

        def get_untargeted_dist(i):
            mzs = self.depth[i-5:i+5,self.center_x-5:self.center_x+5]
            mzs = mzs.flatten()
            mzs = mzs[mzs.nonzero()]
            if len(mzs) != 0 and mzs.mean() != 0:
                mz = mzs.mean()
                return mz
            return 0

        self.targeting = False
        if chosen_j is not None: 
            da_target = self.targets[chosen_j]
            if da_target.min_x <= self.center_x <= da_target.max_x and da_target.avg_mm != -1:
                self.targeting = True
                target_x, target_y = calc_point(
                        da_target.avg_mm, da_target.center_y)

        possible_i = []
        if self.targeting:
            for i in range(self.center_y, 239, 2):
                mth = get_angle(i)
                dist = target_x / math.cos(math.radians(mth))
                x, y = calc_point(dist, i)
                if on_trajectory(self.exit_velocity, self.angle, x, y):
                    possible_i.append((i, dist))
        else:
            for i in range(self.center_y, 239, 2):
                mz = get_untargeted_dist(i)
                if mz != 0:
                    x, y = calc_point(mz, i)
                    if on_trajectory(self.exit_velocity, self.angle, x, y):
                        possible_i.append((i, mz))
        if possible_i:
            i, dist = possible_i[len(possible_i)//2]
            plot_impact(i, dist)

        self.crosshair()

    @property
    def avg_ft(self):
        if self.avg_mm == -1:
            return -1
        return avg_mm / 25.4 / 12

    @property
    def avg_in(self):
        if self.avg_mm == -1:
            return -1
        return avg_mm / 25.4

    def build_targets(self, rects):
        targets = []
        target = None
        for rect, contour in rects:
            (x, y, w, h) = rect
            proposed_target = Target(x, x+w, y, y+h, [contour])
            i = -1
            for j, target in enumerate(targets):
                merged_target = Target.merge(target, proposed_target)
                if self.ok_target(merged_target):
                    targets[j] = merged_target
                    break
            else:
                if len(targets) < self.max_target_count:
                    targets.append(proposed_target)
        return targets

    def ok_target(self, target):
        return (target.max_x - target.min_x < self.max_target_width and
                target.max_y - target.min_y < self.max_target_height)

    def choose_target(self):
        target_dists = []
        for i, target in enumerate(self.targets):
            dist = math.hypot(
                (self.center_x - target.center_x),
                (self.center_y - target.center_y))
            target_dists.append((i, dist))

        target_dists.sort(key=lambda x: x[1])
        if len(target_dists) != 0:
            return target_dists[0][0]

    def crosshair(self, img=None):
        if img is None:
            img = self.contour_img
        corner_thickness = 1
        color = (0, 0, 255)
        if self.targeting:
            color = (255, 0, 255)
        cv2.line(img,
            (self.center_x-10, self.center_y),
            (self.center_x+10, self.center_y),
            color, corner_thickness)
        cv2.line(img,
            (self.center_x, self.center_y-10),
            (self.center_x, self.center_y+10),
            color, 1)

    def on_mouse(self, ev, x, y, flags, userdata):
        if ev == cv2.EVENT_LBUTTONDOWN:
            print ("pixel (%s, %s): " % (x, y))
            print (" depth: %s" % (self.depth[y,x]))
            print (" ir: %s" % (self.ir[y,x]))
            print (" mask: %s" % (self.mask8[y,x]))
            print (" xyz: %s" % (self.xyz[:,y,x]))


def hg_max_apparent_height(theta):
    """
    vision target is 381 mm x 101 mm or smaller
    assume vision target rectangle will not appear rotated more than
    theta degrees, then the max apparent height will be
    """
    theta_r = math.radians(theta)
    return 101 * math.cos(theta_r) + 381 * math.sin(theta_r)
