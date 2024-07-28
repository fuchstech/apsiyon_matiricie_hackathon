# Otopark Yönetimi
Otopark yönetimi iki ana parçaya ayrılarak geliştilirmiştir. Bunlar plaka tanıma sistemi ve otopark doluluk tespitidir. Otopark doluluk tespiti, açık otoparkları olan siteler için hangi aracın hangi park yerinde olduğunu tespit edilebilmesini sağlayan bir sistemdir. Otoparkı gören bir güvenlik kamerasının görüntüsü prototip için replike edilip bu görüntü üzerinde algoritmalar geliştirmiştir. 

#Kurulum
Aşağıdaki adımları izleyerek projeyi kurabilirsiniz:

1. İlk olarak, projeyi klonlayın:
    ```
    git clone https://github.com/apsiyon_hack.git
    ```

2. Ardından, proje dizinine gidin:
    ```
    cd apsiyon_hack
    ```

3. Anaconda ile sanal environment kurun (opsiyonel). Python kurulumu için 5.adıma atlayabilirsiniz :
    ```
    conda create -n apsiyon_matiricie python=3.8 
    ```

4. Conda ortamını çalıştırın:
    ```
    conda activate apsiyon_matiricie
    ```
5. Gerekli paketleri indirin:
    ```
    pip install -r requirements.txt
    ```
5. Otopark kodunu çalıştırın:
    ```
    python otopark/final/main.py
    ```
6. Plaka tanıma kodunu çalıştırın:
    ```
    python plate_recog\main.py
    ```   

Bu adımları takip ederek projeyi başarılı bir şekilde çalıştırabilirsiniz.

- Videonun Tam Hali

[![YouTube Video](https://img.youtube.com/vi/BUE-WcS3bdg/0.jpg)](https://www.youtube.com/watch?v=BUE-WcS3bdg)

<img align=center width=900 src="tanitim.gif" />

# Web
Sosyal klasörü içerisinde yer alan dosyalarda çeşitli resimler, javascript fonksiyonları, HTML5 ve CSS kodları kullanılmıştır. Bu sayfa üzerinde profil oluşturma, aile üyeleri ve evcil hayvan bilgileri ekleme/düzenleme bölümleri bulunmaktadır. Aynı zamanda bu profiller sayesinde komşular tanınarak çeşitli özelliklere göre filtrelenebilecektir. Apartman yönetimi tarafından yapılacak etkinlikler ve olası toplantılar bu sayfada görünecek olup, alıp satılmak istenen eşyaların eklenmesi sağlanmaktadır.

Enerji klasörü içerisinde bulunan dosyalarda javascript fonksiyonları, HTML5 ve CSS kodları kullanılmıştır. Bu sayfada hanenin enerji tüketim grafiği, enerji tüketiminde fayda sağlayacak ve sorulara cevap veren sanal asistan, ortak kullanım alanlarının yoğunluk ve sıcaklıklarını kontrol etmeye ve görmeye yarayan bir ekran bulunmaktadır. Aynı zamanda günlük hava durumunu da yansıtan bir ekran yer almaktadır.

Kullanıcının etkileşime geçeceği uygulamaların tamamı matiricie.com/apsiyon klasörü içerisine de yüklenmiştir.

Canlı kullanım için sırasıyla; matiricie.com/apsiyon/sosyal ve matiricie.com/apsiyon/enerji adresleri ziyaret edilebilir.
