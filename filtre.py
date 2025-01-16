import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


###### INITIALISATION #######

image_path = 'papillon.png'

image_origine = mpimg.imread(image_path)   #recupere l'image d'origine
image = mpimg.imread(image_path)

# Convertir en entiers (0 à 255)
if image.dtype != np.uint8:  # Si les valeurs sont en flottant
    image = (image * 255)  
else:
    image = (image/np.max(image))*255  


# Tronquer l'image
hauteur, largeur = image.shape[:2]
nouvelle_hauteur = (hauteur // 8) * 8
nouvelle_largeur = (largeur // 8) * 8
image = image[:nouvelle_hauteur, :nouvelle_largeur,:]

image = image - 128 #on centralise l'image

taille_image = image.shape

#seuil du filtre
SEUIL=2


#initialisation de P (matrice de passage) avec la formule de la double somme
P=np.zeros((8,8))
for i in range (8):
    for j in range (8):
        if i==0:
            ck=1/math.sqrt(2)
        else:
            ck=1
        P[i,j]= (1/2)*ck*math.cos(((2*j+1)*i*math.pi)/16)



# initialisation de Q : matrice de quantification Q dans la norme de compression JPEG
Q=np.array([[16,11,10,16,24,40,51,61],
            [12,12,13,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]])


    
####### COMPRESSION #########

compressed = np.zeros_like(image, dtype=float)

for c in range(taille_image[2]):  #compression pour tous les canaux de l'image
    for i in range(0, taille_image[0], 8):         #compression par bloc de 8
        for j in range(0, taille_image[1], 8):
            bloc = image[i:i+8, j:j+8,c]
            #application du changement de base
            D = np.dot(P, np.dot(bloc,P.T))    
            D_tilde = np.round(D/Q)
            compressed[i:i+8, j:j+8, c] = D_tilde
            
            # Suppression des coefficients selon le seuil (local au bloc)
            for k in range(8):
                for l in range(8):
                    if k + l >= SEUIL:
                        D_tilde[k, l] = 0
            compressed[i:i+8, j:j+8, c] = D_tilde
                       
                          

####### TAUX DE COMPRESSION #######
nb_coeff_non_zero = np.count_nonzero(compressed)  # Nombre de coefficients non nuls
taux_compression = 100 - ((nb_coeff_non_zero / (taille_image[1]*taille_image[0]*3)) * 100 ) 
print(f"taux de compression : {taux_compression}")


####### DECOMPRESSION ###########
decompressed = np.zeros_like(image, dtype=np.float32)

for c in range(taille_image[2]):  #decompression pour tous les canaux
    for i in range(0, taille_image[0], 8):     #decompression par blocs de 8
        for j in range(0, taille_image[1], 8):
            B = compressed[i:i+8,j:j+8,c]
            #opérations inverse à la compression
            D_tilde = B * Q
            D = np.dot(P.T, np.dot(D_tilde,P))
            D=np.round(D)
            decompressed[i:i+8, j:j+8, c] = D +128
         
            
decompressed = np.clip(decompressed, 0, 255).astype(np.uint8)   #permet de rentrer toutes les valeurs entre 0 et 255

###### CALCUL DE L'ERREUR ########
erreur = (np.linalg.norm(((image+128)-decompressed))/(np.linalg.norm(image))) *100
print(f"l'erreur est : {erreur}")

fig,axes = plt.subplots(2,2)
axes[0, 0].imshow(image_origine)  
axes[0, 0].set_title("Image d'origine")  
axes[0, 1].imshow(compressed)  
axes[0, 1].set_title("Image compressée avec filtre")
axes[1, 0].imshow(decompressed)  
axes[1, 0].set_title("Image décompressée")
axes[1,1].axis('off')
plt.tight_layout()
plt.show()


