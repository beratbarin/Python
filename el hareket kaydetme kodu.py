import cv2                          # Karşılaştırma için kullanılacak veriler bu kod ile oluşturulur.
import numpy as np
import os

Kamera = cv2.VideoCapture(0)
kernel = np.ones((12,12),np.uint0)

isim = "isim"                      # Oluşturulacak verinin ismi tanımlanır.

while True:
    ret, Kare = Kamera.read()
    Kesilmis_Kare = Kare[0:300,0:300]
    Kesilmis_Kare_HSV = cv2.cvtColor(Kesilmis_Kare,cv2.COLOR_BGR2HSV)

    Alt_Degerler = np.array([0,30,60])
    Ust_Degerler = np.array([40,255,255])

    Renk_filtresi_sonucu = cv2.inRange(Kesilmis_Kare_HSV,Alt_Degerler,Ust_Degerler)
    Renk_filtresi_sonucu = cv2.morphologyEx(Renk_filtresi_sonucu,cv2.MORPH_CLOSE,kernel)
    Sonuc = Kesilmis_Kare.copy()

    cnts,_ = cv2.findContours(Renk_filtresi_sonucu,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    Max_Genıslık = 0
    Max_Uzunluk = 0
    Max_ındex = -1

    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h = cv2.boundingRect(cnt)
        if (w>Max_Genıslık and h>Max_Uzunluk):
            Max_Uzunluk = h
            Max_Genıslık = w
            Max_ındex = t

    if (len(cnts)>0):
        x,y,w,h = cv2.boundingRect(cnts[Max_ındex])
        cv2.rectangle(Sonuc, (x,y), (x + w, y + h), (0, 255, 0), 2)
        El_Resim = Renk_filtresi_sonucu[y:y+h,x:x+w]
        cv2.imshow("El Resim", El_Resim)


    cv2.imshow("Sonuc",Sonuc)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite("Veri/"+isim+".jpg",El_Resim)              # Program çalıştıktan sonra veri oluşturmak için El_Resim görüntü çıktısı kulllanılır.
Kamera.release()                                       # El_Resim de nasıl bir görüntü var ise "q" tuşuna bastığımızda o görüntü Veri klasörüne eklenir.
cv2.destroyAllWindows()                                # Ve program sonlanır.

