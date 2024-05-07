import numpy as np
import cv2

class Color_Composing_Machine:
    def __init__(self):
        self.R_face = np.ones((16, 16), dtype=np.uint8)
        self.G_face = np.zeros((16, 16), dtype=np.uint8)
        self.B_face = np.zeros((16, 16), dtype=np.uint8)
        self.R_face_image = np.zeros((320, 320, 3), dtype=np.uint8)
        self.G_face_image = np.zeros((320, 320, 3), dtype=np.uint8)
        self.B_face_image = np.zeros((320, 320, 3), dtype=np.uint8)
        self.RG_face = np.zeros((320, 320, 3), dtype=np.uint8)
        self.GB_face = np.zeros((320, 320, 3), dtype=np.uint8)
        self.BR_face = np.zeros((320, 320, 3), dtype=np.uint8)
        self.state = "start"
        self.rows_count = {"R": [0] * 16, "G": [0] * 16, "B": [0] * 16}
        self.available_rows = {"R": list(range(16)), "G": list(range(16)), "B": list(range(16))}
        self.raw_status_R={i:0 for i in range(16)}
        self.raw_status_G={i:0 for i in range(16)}
        self.raw_status_B={i:0 for i in range(16)}

    def restore_available_raws(self):
        self.raw_status_R={i:0 for i in range(16)}
        self.raw_status_G={i:0 for i in range(16)}
        self.raw_status_B={i:0 for i in range(16)}
        self.available_rows = {"R": list(range(16)), "G": list(range(16)), "B": list(range(16))}

    
    def input_face_coordinates(self, face):
        while True:
            x = int(input(f"Enter x coordinate for {face}-face (0-15): "))
            y = int(input(f"Enter y coordinate for {face}-face (0-15): "))
            z = int(input(f"Enter z coordinate for {face}-face (0-15): "))
            if not self.available_rows["R"] or not self.available_rows["G"] or not self.available_rows["B"]:
                self.restore_available_raws
            if 0 <= x <= 15 and 0 <= y <= 15 and 0 <= z <= 15 :
                if self.raw_status_R[x]>=3:
                    print(f"the raw number {x} is not available in R")
                    self.available_rows["R"].remove(x)
                    self.raw_status_R[x]=0
                    print(self.available_rows["R"])
                    continue
                elif self.raw_status_G[y]==3:
                    print(f"the raw number {y} is not available in G")
                    self.available_rows["G"].remove(y)
                    self.raw_status_G[y]=0
                    print(self.available_rows["G"])
                    continue
                elif self.raw_status_B[z]==3:
                    print(f"the raw number {y} is not available in B")
                    self.available_rows["B"].remove(y)
                    self.raw_status_B[z]=0
                    print(self.available_rows["B"])
                    continue
                self.raw_status_B[z]+=1
                self.raw_status_G[y]+=1
                self.raw_status_R[x]+=1
                self.available_rows["R"].append(x) if x not in self.available_rows["R"] else None
                self.available_rows["G"].append(y) if y not in self.available_rows["G"] else None
                self.available_rows["B"].append(z) if z not in self.available_rows["B"] else None
                return x, y, z
            else:
                print("Invalid coordinates. Please enter coordinates in the range 0-15.")

    def input_rotation_angle(self,face):
        while True:
            angle = int(input(f"Enter rotation angle (0, 90, 180, 270) for {face}: "))
            if angle in [0, 90, 180, 270]:
                return angle
            else:
                print("Invalid rotation angle. Please enter 0, 90, 180, or 270.")


    def pixelate_RG_face(self, x, y, z):
        for p in range(16):
            for q in range(16):
                h = self.R_face[x, p]
                k = self.G_face[y, q]
                self.RG_face[p*20:p*20+19, q*20:q*20+19, 0] = 0
                self.RG_face[p*20:p*20+19, q*20:q*20+19, 1] = k 
                self.RG_face[p*20:p*20+19, q*20:q*20+19, 2] = h 
        # self.display_image(self.RG_face)
    def pixelate_GB_face(self, x, y, z):
        for q in range(16):
                for r in range(16):

                    k = self.G_face[y, q]
                    l = self.B_face[x, r]
                    self.GB_face[q*20:q*20+19, r*20:r*20+19, 0] = l
                    self.GB_face[q*20:q*20+19, r*20:r*20+19, 1] = k 
                    self.GB_face[:,:, 2] = 0
        # self.display_image(self.GB_face)  

    def pixelate_BR_face(self, x, y, z):
        for r in range(16):
            for p in range(16):
                h = self.R_face[x, p]
                l = self.B_face[z, r]
                self.BR_face[ r*20:r*20+19, p*20:p*20+19,  0] = l  
                self.BR_face[:, :, 1] = 0  
                self.BR_face[r*20:r*20+19,p*20:p*20+19,  2] = h
        # self.display_image(self.BR_face) 
    def make_face_image(self,matrix,k):

        img=np.zeros((320, 320, 3), dtype=np.uint8)
        for i in range(16):
            for j in range(16):
                img[i*20:i*20+19, j*20:j*20+19,k]=matrix[i][j]
        return img

                
    


    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    # def display_image(self, image):
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)


    def process_input(self):
        total_charge = 0
        n=0
        while total_charge < 20:
            total_charge = int(input("Enter the price you want to pay (>= 20): "))
            if total_charge < 20:
                print("Price must be at least 20.")

        if total_charge >= 60:
            self.state = "show_all_faces"
        elif total_charge < 60:
            self.state = "select_faces"
            n=2 if total_charge>40 else 1

        return total_charge,n

    def run(self):
        while True:
            if self.state == "start":
                total_charge,n = self.process_input()

            elif self.state == "show_all_faces":

                cv2.imshow("RG phase",self.RG_face)
                cv2.imshow("GB phase",self.GB_face)
                cv2.imshow("BR phase",self.BR_face)
                cv2.waitKey(1)

                change = total_charge - 60
                if change > 0:
                    print(f"Change: {change}")
                cv2.destroyAllWindows
                break

            elif self.state == "select_faces":
                for i in range(n):
                    choice = input(f"Enter the faces{(i+1)} you want to show (e.g., RG, GB, BR): ")
                    if choice == "RG":
                        # self.pixelate_RG_face(x,y,z)
                        cv2.imshow("RG phase",self.RG_face)
                    elif choice == "GB":
                        cv2.imshow("GB phase",self.GB_face)
                    elif choice == "BR":
                        # self.pixelate_BR_face(x,y,z)
                        cv2.imshow("BR phase",self.BR_face)

                change = total_charge - 20*n
                if change > 0:
                    print(f"Change: {change}")
                cv2.waitKey(1)
                # cv2.destroyAllWindows
                break

c_r,c_g,c_b = 0, 0, 0  

machine = Color_Composing_Machine()

machine.R_face = np.random.permutation(np.arange(256)).reshape((16, 16))
machine.G_face = np.random.permutation(np.arange(256)).reshape((16, 16))
machine.B_face = np.random.permutation(np.arange(256)).reshape((16, 16))
machine.R_face_image=machine.make_face_image(machine.R_face,2)
machine.G_face_image=machine.make_face_image(machine.G_face,1)
machine.B_face_image=machine.make_face_image(machine.B_face,0)
cv2.imshow("R phase",machine.R_face_image)
cv2.imshow("G phase",machine.G_face_image)
cv2.imshow("B phase",machine.B_face_image)
cv2.waitKey(50)


while True:

    c_r, c_g, c_b = machine.input_face_coordinates("RG"), machine.input_face_coordinates("GB"), machine.input_face_coordinates("BR")
    angle_rg = machine.input_rotation_angle(("RG"))
    angle_gb = machine.input_rotation_angle("GB")
    angle_br = machine.input_rotation_angle("BR")

    machine.pixelate_RG_face(*c_r)
    machine.RG_face = machine.rotate_image(machine.RG_face, angle_rg)
    machine.pixelate_GB_face(*c_g)
    machine.GB_face = machine.rotate_image(machine.GB_face, angle_gb)

    machine.pixelate_BR_face(*c_b)
    machine.BR_face = machine.rotate_image(machine.BR_face, angle_br)

    machine.run()
    
    if input("Do you want to continue (y/n)? ").lower() != 'y':
        cv2.destroyAllWindows
        break
    else:
        machine.state="start"
