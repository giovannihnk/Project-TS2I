import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


###### INITIALISATION #######

image_path = 'papillon.png'

image = mpimg.imread(image_path)
if image.dtype != np.uint8:  # Si les valeurs sont en flottant (0 à 1)
    image = (image * 255).astype(np.uint8)  # Convertir en entiers (0 à 255)


hauteur, largeur = image.shape[:2]
nouvelle_hauteur = (hauteur // 8) * 8
nouvelle_largeur = (largeur // 8) * 8

image = image[:,:,0]  #on garde qu'une seule couleur (2D)


# Tronquer l'image
image = image[:nouvelle_hauteur, :nouvelle_largeur]
taille_image = image.shape
#print(image)
# plt.imshow(image)
# plt.show()
#print(image.shape)


#initialisation de P (matrice de passage) avec la formule de la double somme
P=np.zeros((8,8))
for i in range (8):
    for j in range (8):
        if i==0:
            ck=1/math.sqrt(2)
        else:
            ck=1
        P[i,j]= (1/2)*ck*math.cos(((2*j+1)*i*math.pi)/16)
#print(P)


####### COMPRESSION #########


# initialisation de Q : matrice de quantification Q dans la norme de compression JPEG
Q=np.array([[16,11,10,16,24,40,51,61],
            [12,12,13,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]])



def compression(M,P,Q,taille):
    for i in range(0, taille[0], 8):
        for j in range(0, taille[1], 8):
            D = P @ M[i:i+8, j:j+8] @ P.T
            D_tilde = np.divide(D,Q)
        #print(D)
        # prendre la partie entière
            np.floor(D_tilde)
            M[i:i+8, j:j+8] = D_tilde
        
    return M

img2 = compression(image,P,Q,taille_image)
#  — Compter le nombre de cœfficients non nuls pour obtenir le taux de compression
nb_coeff_non_zero = np.count_nonzero(img2)  # Nombre de coefficients non nuls
taux_compression = 100 - ((nb_coeff_non_zero / (taille_image[1]*taille_image[0])) * 100 ) 
print(f"taux de compression : {taux_compression}")


plt.imsave('image_recomposee2.png',img2)
image_recomposee2 = mpimg.imread('image_recomposee2.png')
plt.imshow(image_recomposee2)
plt.show()


def decompression(M,P,Q,taille):
    transpose = P.T
    for i in range(0, taille[0], 8):
        for j in range(0, taille[1], 8):
            C = M[i:i+8, j:j+8] * Q
            M[i:i+8, j:j+8] = transpose @ C @ P
    return M



img3 = decompression(img2,P,Q,taille_image)
# print(img2)
plt.imsave('image_recomposee3.png',img3)
image_recomposee3 = mpimg.imread('image_recomposee3.png')
plt.imshow(image_recomposee3)
plt.show()
