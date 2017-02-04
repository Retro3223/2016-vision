import math
import pygrip
import cv2
import numpy
from networktables import NetworkTable
from numpy_pool import NumpyPool
from best_fit_line import BestFitLine
from utils import (
    into_uint8,
    into_uint16_mask,
    minAreaBox,
    boxCenter,
)
from data_logger import DataLogger, Replayer
from xyz_converter import (
    depth_to_xyz,
    x_mm_to_pixel,
    x_pixel_to_mm,
    distance,
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
        self.hg_right_edge = None
        self.hg_left_edge = None
        self.hg_sees_target = False
        self.hg_x_offset_pixel = 100000
        self.hg_y_offset_pixel = 100000
        self.mode = 0
        self.area_threshold = 10
        self.contours = []
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

    def setup_mode_listener(self):
        self.sd.addTableListener(self.value_changed)

    def value_changed(self, table, key, value, is_new):
        if key == "structureMode":
            if value in [0, 1, 2, 3, 4, 5]:
                self.set_mode(value)

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
            return self.display

    def hg_publish(self):
        if len(self.contours) == 0: 
            self.sd.putBoolean("seesHighGoal", False)
            return
        rect = cv2.boundingRect(self.contours[0])
        center_i = rect[0] + rect[2] // 2
        center_j = rect[1] + rect[3] // 2
        x = self.xyz[0, center_j, center_i]
        x_pixel_offset = 160 - center_i
        self.sd.putBoolean("seesHighGoal", True)
        #self.sd.putNumber("xOffsetHighGoal", self.hg_x_offset_mm)
        self.sd.putNumber("xOffsetHighGoal", self.hg_x_offset_pixel)
        #self.sd.putNumber("xPixelOffsetHighGoal", self.hg_x_offset_pixel)
        self.sd.putNumber("yPixelOffsetHighGoal", self.hg_y_offset_pixel)
        self.sd.putNumber("zOffsetHighGoal", self.hg_z_offset_mm)
        self.sd.putNumber("thetaHighGoal", self.hg_theta)
    
    def hg_draw_hud(self):
        if self.mode == 2:
           z0 = 2000
           cr = 8 / 123
           z0 = int(z0 * cr)
           x0 = 0
           center = (320//2,240//2)
           cv2.circle(self.display,center,4,(0,255,0),1)
           if len(self.contours) ==0 : return
           contour = self.contours[0]
           a = cv2.boundingRect(contour) 
           b = ((a[1]+a[1]+a[3])//2,(a[0]+a[0]+a[2])//2)
           x_goal = self.xyz[0,b[0],b[1]]
           z_goal = self.xyz[2,b[0],b[1]]
           if abs(x_goal) > 2460 : return 
           else : i =int( 160 + (x_goal * 8 // 123))
           if abs(z_goal) > 3690 : return 
           else: j = 240 - int((z_goal * 8 // 123))
           cv2.circle(self.display,(i,j),17,(0,255,123),1)
           #print(b,x_goal,z_goal,i,j)
           #cv2.circle(self.display,b,17,(0,0,255),1)
           
    def process_depths(self):
        """
        """
        self.display_depth()
        self.display_raw_ir()

        if self.is_hg_position:
            if self.mode == DISP_IR_MASK3:
                self.display[:] = 0
            self.hg_mask_shiny()
            depth_to_xyz(depth=self.depth, xyz=self.xyz)
            self.hg_filter_shiniest()
            self.hg_find_edges()
            self.hg_make_target()
            self.hg_draw_hud()
            self.hg_publish()
        else:
            pass

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

        def get_center_xyz(c):
            box = minAreaBox(c)
            center_pixel = boxCenter(box)
            xyz = self.xyz[:, center_pixel[1], center_pixel[0]]
            return xyz

        if len(contours) > 1:
            center_xyz0 = get_center_xyz(contours[0])
            contours = [c for c in contours if distance(get_center_xyz(c), center_xyz0) < 700]
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
            # bugger, I was using the wrong xyz converter. need to revisit!
            width = abs(x_part.max() - x_part.min())
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
        depth_part = self.depth[y:y+h, x:x+w]
        mask_part[depth_part == 0] = 0
        mid_depth = numpy.median(depth_part[mask_part == 255])
        self.pool.release_gray(mask)
        return mid_depth

    def display_edges(self, edges, mid_points):
        if self.mode == DISP_EDGES:
            temp = self.pool.get_gray()
            into_uint8(self.depth, dst=temp)
            cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR, dst=self.display)
            temp[:] = 0
            for contour in self.contours:
                mid_depth = self.hg_median_dist(contour)
                # ignore pixels with outlier distances 
                ixs = self.depth <= mid_depth + 200
                ixs &= self.depth >= mid_depth - 200
                temp[ixs] = 155
            if len(self.contours) != 0:
                c = self.contours[0]
                (x,y,w,h) = cv2.boundingRect(c)
                i = x+w//2 
                j = y+h//2
                if self.depth[j, i] != 0:
                    cv2.floodFill(temp, None, (i, j), 255, 1, 1)
                    ixs = temp == 255
                    self.display[ixs,:] = [97,206,202]
            self.pool.release_gray(temp)
            for contour in self.contours:
                box = minAreaBox(contour)
                self.draw_box(self.display, box)
            #for edge in edges:
            #    cv2.circle(self.display, edge, 2, (255, 0, 255), -1)
            #for pt in mid_points:
            #    cv2.circle(self.display, pt, 2, (255, 255, 255), -1)

            def draw_edge(edge: BestFitLine):
                if edge is not None: 
                    t0 = edge.t_from_y(0)
                    x0 = edge.x_from_t(t0)
                    tn = edge.t_from_y(239)
                    xn = edge.x_from_t(tn)
                    pt1 = (int(x0), 0)
                    pt2 = (int(xn), 239)
                    cv2.line(self.display, pt1, pt2, (255, 0, 255), 1)
            
            #draw_edge(self.hg_left_edge)
            #draw_edge(self.hg_right_edge)

    def draw_box(self, img, box):
        pt1 = (box[0])
        pt2 = (box[1])
        pt3 = (box[2])
        pt4 = (box[3])
        cv2.line(img, pt1, pt2, (0, 0, 255), 1)
        cv2.line(img, pt2, pt3, (0, 0, 255), 1)
        cv2.line(img, pt3, pt4, (0, 0, 255), 1)
        cv2.line(img, pt1, pt4, (0, 0, 255), 1)

    def hg_find_edges(self):
        depth_a = self.depth[:, 0:-1]
        depth_b = self.depth[:, 1:]
        depth_diff = numpy.absolute((depth_a - depth_b).astype('int16'))
        idx = depth_diff < self.area_threshold * 10
        depth_diff[idx] = 0
        depth8 = self.pool.get_gray()
        depth8[:, :] = 255
        depth8[idx] = 0
        right_edges = []
        left_edges = []
        mid_points = []
        def pixel_x(pt):
            return pt[0]

        for contour in self.contours:
            box = minAreaBox(contour)
            mid_x, mid_y = boxCenter(box)
            mid_points.append((mid_x, mid_y))
            # box[i] is a seq (i, j) where i is 320 dir, j is 240
            pts1 = (box[0], box[1])
            pts2 = (box[2], box[3])
            dx1 = abs(pixel_x(box[0]) - pixel_x(box[1]))
            dx2 = abs(pixel_x(box[1]) - pixel_x(box[2]))
            # assume bounding box is rectangle with longer sides horizontal
            # pts1 and pts2 should be the horizontal sides
            if dx1 < dx2:
                pts1 = (box[0], box[3])
                pts2 = (box[2], box[1])

            redges1 = [self.hg_find_right_edge(*pt) for pt in [pts1, pts2]]
            right_edges.extend([x for x in redges1 if x is not None])
            ledges1 = [self.hg_find_left_edge(*pt) for pt in [pts1, pts2]]
            left_edges.extend([x for x in ledges1 if x is not None])
        all_edges = []
        all_edges.extend(right_edges)
        all_edges.extend(left_edges)
        self.hg_right_edge = None
        self.hg_left_edge = None
        if len(right_edges) > 1:
            self.hg_right_edge = BestFitLine(right_edges)
            if abs(self.hg_right_edge.xy_slope()) < 2.0:
                self.hg_right_edge = None
        
        if len(left_edges) > 1:
            self.hg_left_edge = BestFitLine(left_edges)
            if abs(self.hg_left_edge.xy_slope()) < 2.0:
                self.hg_left_edge = None
        
        self.display_edges(all_edges, mid_points)
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

        start_pt = (min(239, (pt1[0]+pt2[0])//2), pt2[1])
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

        start_pt = (min(239, (pt1[0]+pt2[0])//2), pt1[1])
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

    def hg_make_target(self):
        edge_adjust = (
            (self.hg_right_edge is not None) ^ 
            (self.hg_left_edge is not None)
        )

        if len(self.contours) == 0:
            self.hg_sees_target = False
            return

        self.hg_sees_target = True
        contour = self.contours[0]
        box = minAreaBox(contour)
        (cx_pixel, cy_pixel) = boxCenter(box)
        dist_mm = self.hg_median_dist(contour)
        # todo: deal with sensor incline
        self.hg_z_offset_mm = dist_mm
        self.hg_x_offset_mm = -self.xyz[0, cy_pixel, cx_pixel]
        if self.hg_x_offset_mm == 0.0:
            self.hg_x_offset_mm = -x_pixel_to_mm(cx_pixel, dist_mm)

        edge_adjust = False #turned off for now, is buggy
        if edge_adjust and self.hg_left_edge is not None:
            t = self.hg_left_edge.t_from_y(cy_pixel)
            x = int(self.hg_left_edge.x_from_t(t))
            while abs(self.depth[cy_pixel, x] - dist_mm) > 200 and x != cx_pixel:
                x += 1

            x_offset_edge = self.xyz[0, cy_pixel, x]
            # todo: be less lazy and don't assume the edge's slope is parallel
            # with pixel y axis
            self.hg_x_offset_mm = -x_offset_edge - 170
        elif edge_adjust and self.hg_right_edge is not None:
            t = self.hg_right_edge.t_from_y(cy_pixel)
            x = int(self.hg_right_edge.x_from_t(t))
            while abs(self.depth[cy_pixel, x] - dist_mm) > 200 and x != cx_pixel:
                x -= 1

            x_offset_edge = self.xyz[0, cy_pixel, x]
            self.hg_x_offset_mm = -x_offset_edge + 170

        # we /shouldn't/ have been able to get here if z offset = 0..
        if self.hg_z_offset_mm == 0:
            self.hg_theta = 0
        else:
            self.hg_theta = math.atan(self.hg_x_offset_mm  / self.hg_z_offset_mm)

        print (cx_pixel)
        self.hg_x_offset_pixel = 160 - cx_pixel
        self.hg_y_offset_pixel = 120 - cy_pixel
        if self.mode == DISP_EDGES:
            px = x_mm_to_pixel(-self.hg_x_offset_mm, dist_mm)
            print (cx_pixel, self.hg_x_offset_pixel)
            #print (edge_adjust, cx_pixel, cy_pixel, self.hg_x_offset_mm)
            #print (self.hg_z_offset_mm)
            #print ("  ", px)
            cv2.circle(self.display, (px, cy_pixel), 2, (0, 0, 255), 2)

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


