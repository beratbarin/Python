import cv2
import numpy as np                                                       # Opencv, numpy kütüphaneleri ve os modülü çalıştırılir.
import os

Kamera = cv2.VideoCapture(0)                                             # Bu fonksiyon ile hod çalıştığı zaman kamera açılır.
kernel = np.ones((12,12),np.uint0)                                       # Verilen değerlerde görüntü matris oluşturulur.

def ResimFarkBul(Resim1,Resim2):                                                         # Fonksiyon yaratılır.
    Resim2 = cv2.resize(Resim2,(Resim1.shape[1],Resim1.shape[0]))                        # Bu metod ile görüntünün YxG i boyutlandırılır.
    Fark_Resim = cv2.absdiff(Resim1,Resim2)                                              # İki görüntü matrisi arasında çıkarma işlemi yapılır, hareketli kısımlar gösterilir.
    Fark_Sayi = cv2.countNonZero(Fark_Resim)               # Görüntünün tamamen siyah olup olmadığına karar vermek için kullanılabilen sıfır olmayan piksel sayısı elde edilir.
    return Fark_Sayi

def VeriYükle():
    Veri_Isimler = []                                                            # El hareket tanımı için gereken veriler için fonksiyon yaratılır.
    Veri_Resimler = []

    Dosyalar = os.listdir("Veri/")                                               # Kullanılacak verilerin hangi klasörde olduğu gösterilir.
    for Dosya in Dosyalar:
        Veri_Isimler.append(Dosya.replace(".jpg",""))                            # Klosördeki .jpg ile biten dosyalar dizinlenir.
        Veri_Resimler.append(cv2.imread("Veri/"+Dosya,0))                        # Dizinlenen dosyalar okunur
    return Veri_Isimler,Veri_Resimler

def Sınıflandır(Resim,Veri_Isimler,Veri_Resimler):
    Min_index = -1                                                                 # Veri boyutları ile alınacak görüntünün boyutları hizalanır. Başlangıç değerleri yazılır.
    Min_Deger = ResimFarkBul(Resim,Veri_Resimler[0])
    for t in range(len(Veri_Isimler)):                                             # Dizi oluşturulur ve len() fonk. ile eleman sayısı tespit edilir.
        Fark_Deger = ResimFarkBul(Resim,Veri_Resimler[t])
        if(Fark_Deger<Min_Deger):                                                  # Veriler ile elimizin görüntüsü karşılaştırılır ve tanım için bağ kurulmuş olur.
            Min_Deger = Fark_Deger
            Min_index = t
    return Veri_Isimler[Min_index]

Veri_Isimler,Veri_Resimler = VeriYükle()                                           # İşlem bittikten sonra veriler kullanıma hazırdır.


while True:
    ret, Kare = Kamera.read()                                                                # Orjinal kamera görüntüsü okunur ve değişkene(Kare) atanır.
    Kesilmis_Kare = Kare[0:250,0:250]                                              # Okunan görüntü verilen değerler doğrultusunda biçimlenir. Daha özçekim bir görüntü oluşur.
    Kesilmis_Kare_HSV = cv2.cvtColor(Kesilmis_Kare,cv2.COLOR_BGR2HSV)  # Biçimlenen görüntü filtrelenir. Seçilen filtre el tespiti için idealdir.(Nesne beyaz arka plan siyah)

    Alt_Degerler = np.array([0,30,60])                                             # Filtreleme ayarları düzenlenir. Değerlerde değişiklik yapılabilir.
    Ust_Degerler = np.array([40,255,255])

    Renk_filtresi_sonucu = cv2.inRange(Kesilmis_Kare_HSV,Alt_Degerler,Ust_Degerler)                          # Girilen ayarlar görüntüye işlenir.
    Renk_filtresi_sonucu = cv2.morphologyEx(Renk_filtresi_sonucu,cv2.MORPH_CLOSE,kernel)                     # Görüntüdeki nesnelerin ana hatları belirlenir.
    Sonuc = Kesilmis_Kare.copy()                                                                             # Görüntünün(Kesilmis_Kare) yapısı yeni bir görüntüye uyarlanır.
                                                                                 # Ön planda kalacak elimizdeki küçük delikleri ve ya üzerindeki küçük siyah noktaları kapatır.
    cnts,_ = cv2.findContours(Renk_filtresi_sonucu,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    Max_Genıslık = 0
    Max_Uzunluk = 0                                                # Aynı renk ve yoğunluğa sahip olan tüm kesintisiz pikseller bulunur.(Kontur) Başlangıç değerleri verilir.
    Max_ındex = -1

    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h = cv2.boundingRect(cnt)                                           # El tespiti için gereken konturlar bulunduktan yerleri hesaplanır, tespit edilir.
        if (w>Max_Genıslık and h>Max_Uzunluk):
            Max_Uzunluk = h                                                       # El hareketlerindeki değişimler boyutlarda yeni değerlere yol açar. Bu kısımda bu sağlanır.
            Max_Genıslık = w
            Max_ındex = t

    if (len(cnts)>0):
        x,y,w,h = cv2.boundingRect(cnts[Max_ındex])                    # Konturların tespiti yapılınca elimiz artık görüntüde ayırt edilebilir haldedir. Hesaplamalar işlenir.
        cv2.rectangle(Sonuc, (x,y), (x + w, y + h), (0, 255, 0), 2)                     # El hareket değişimlerine göre Sonuc görüntüsünde elimiz dikdörtgen içerisine alınır.
        El_Resim = Renk_filtresi_sonucu[y:y+h,x:x+w]       # Filtrelenmiş görüntü elimizin hareketine göre değişen dikdörtgen çizimin şeklini alan yeni bir görüntüye dönüşür.
        cv2.imshow("El Resim", El_Resim)                                                  # Bu yeni görüntünün(El_Resim) çıktısı alınır.
        print(Sınıflandır(El_Resim,Veri_Isimler,Veri_Resimler))              # Veriler ile el görüntülerin karşılaştırma sonuçları terminale sonsuz bir döngü şeklinde yazılır.




    cv2.imshow("Kare",Kare)                                               # Görüntünün(Kesilmis_Kare) yapısı kopyalanır.(Değişikler hariç)
    cv2.imshow("Kesilmis Kare",Kesilmis_Kare)                                           # Biçimlenmiş orjinal görüntünün(Kesilmis_Kare) çıktısı alınır.
    cv2.imshow("Renk Filtresi Sonucu",Renk_filtresi_sonucu)               # Üzerinde el tespiti için ideal filtreleme yapılmış görüntünün(Renk_filtresi_sonucu) çıktısı alınır.
    cv2.imshow("Sonuc",Sonuc)                                # Elimizin hareketleri tespit edip boyut değiştiren dikdörtgen çizimin bulunduğu görüntünün(Sonuc) çıktısı alınır.

    if cv2.waitKey(1) & 0xFF == ord('q'):                               # Program çalıştıktan sonra "q" tuşu ile sonlandırırız.
        break

Kamera.release()                                                        # Program sonlandıktan sonra kamera ve tüm pencereler kapatılır.
cv2.destroyAllWindows()

