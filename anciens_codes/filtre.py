import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time 

###### INITIALISATION #######

image_path = 'papillon.jpeg' #chemin de l'image 

#lecture de l'image
image_origine = mpimg.imread(image_path)  
image = mpimg.imread(image_path)

#Conversion des valeurs entre 0 et 255
if image.dtype != np.uint8:  # Si les valeurs sont en flottant (0 à 1)
    image = (image * 255) 
else:
    image = (image/np.max(image))*255



# On tronque l'image avec ses nouvelles dimensions
hauteur, largeur = image.shape[:2]
nouvelle_hauteur = (hauteur // 8) * 8
nouvelle_largeur = (largeur // 8) * 8
image = image[:nouvelle_hauteur, :nouvelle_largeur,:]

#centralisation des valeurs de la matrice
image = image - 128

taille_image = image.shape

#seuil du filtre
SEUIL=6


#initialisation de P (matrice de passage) avec la formule de la double somme
P=np.zeros((8,8))
for i in range (8):
    for j in range (8):
        if i==0:
            ck=1/math.sqrt(2)
        else:
            ck=1
        P[i,j]= (1/2)*ck*math.cos(((2*j+1)*i*math.pi)/16)



# initialisation de Q : matrice de quantification Q 
Q=np.array([[16,11,10,16,24,40,51,61],
            [12,12,13,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]])



####### COMPRESSION #########

compressed = np.zeros_like(image, dtype=float) #initialisation d'une matrice pour stocker les coefficients de compression

start_compression = time.time() # début du chronomètre pour le temps d'exécution

for c in range(taille_image[2]):  #compressions pour tous les canaux
    for i in range(0, taille_image[0], 8):   #compression par bloc de 8x8
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
                       
end_compression = time.time() # arrêt du chronomètre

time_taken_compression = end_compression - start_compression # calcul du temps d'exécution de la compression
print(f"Temps d'exécution de la compression : {time_taken_compression:.6f} secondes") #afficher le temps d'exécution 


####### DECOMPRESSION ###########
decompressed = np.zeros_like(image, dtype=np.float32)

start_decompression = time.time() #début du chronomètre pour le temps d'exécution

for c in range(taille_image[2]):
    for i in range(0, taille_image[0], 8):
        for j in range(0, taille_image[1], 8):
            B = compressed[i:i+8,j:j+8,c]
            #opérations inverse à la compression
            D_tilde = B * Q
            D = np.dot(P.T, np.dot(D_tilde,P))
            D=np.round(D)
            decompressed[i:i+8, j:j+8, c] = D +128

#on limite les valeurs de la matrice entre 0 et 255
decompressed = np.clip(decompressed, 0, 255).astype(np.uint8)  

end_decompression = time.time() #arrêt du chronomètre

time_taken_decompression = end_decompression - start_decompression # calcul du temps d'exécution de la décompression
print(f"Temps d'exécution de la decompression : {time_taken_decompression:.6f} secondes") #affichage du temps d'exécution



######## CALCUL DU TAUX DE COMPRESSION ########
nb_coeff_non_zero = np.count_nonzero(compressed)  # Nombre de coefficients non nuls
taux_compression = 100 - ((nb_coeff_non_zero / (taille_image[1]*taille_image[0]*3)) * 100 ) 
print(f"taux de compression : {taux_compression}")


###### CALCUL DE L'ERREUR ########
erreur = (np.linalg.norm(((image+128)-decompressed))/(np.linalg.norm(image))) *100
print(f"l'erreur est : {erreur}")

##### AFFICHAGE DES IMAGES
fig,axes = plt.subplots(2,2)
axes[0, 0].imshow(image_origine)  
axes[0, 0].set_title("Image d'origine")  
axes[0, 1].imshow(compressed)  
axes[0, 1].set_title("Image compressée")
axes[1, 0].imshow(decompressed)  
axes[1, 0].set_title("Image décompressée")
axes[1,1].axis('off')
plt.tight_layout()
plt.show()




