import cv2
import numpy as np
import os

Kamera = cv2.VideoCapture(0)
kernel = np.ones((12,12),np.uint0)

def ResimFarkBul(Resim1,Resim2):
    Resim2 = cv2.resize(Resim2,(Resim1.shape[1],Resim1.shape[0]))
    Fark_Resim = cv2.absdiff(Resim1,Resim2)
    Fark_Sayi = cv2.countNonZero(Fark_Resim)
    return Fark_Sayi

def VeriYükle():
    Veri_Isimler = []
    Veri_Resimler = []

    Dosyalar = os.listdir("Veri/")
    for Dosya in Dosyalar:
        Veri_Isimler.append(Dosya.replace(".jpg",""))
        Veri_Resimler.append(cv2.imread("Veri/"+Dosya,0))
    return Veri_Isimler,Veri_Resimler

def Sınıflandır(Resim,Veri_Isimler,Veri_Resimler):
    Min_index = -1
    Min_Deger = ResimFarkBul(Resim,Veri_Resimler[0])
    for t in range(len(Veri_Isimler)):
        Fark_Deger = ResimFarkBul(Resim,Veri_Resimler[t])
        if(Fark_Deger<Min_Deger):
            Min_Deger = Fark_Deger
            Min_index = t
    return Veri_Isimler[Min_index]

Veri_Isimler,Veri_Resimler = VeriYükle()


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
        print(Sınıflandır(El_Resim,Veri_Isimler,Veri_Resimler))




    cv2.imshow("Kare",Kare)
    cv2.imshow("Kesilmis Kare",Kesilmis_Kare)
    cv2.imshow("Renk Filtresi Sonucu",Renk_filtresi_sonucu)
    cv2.imshow("Sonuc",Sonuc)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Kamera.release()
cv2.destroyAllWindows()

