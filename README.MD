# Tanıtım Videomuz

<div align="center">
  <a href="https://www.youtube.com/watch?v=BUE-WcS3bdg" target="_blank">
    <img src="https://img.youtube.com/vi/BUE-WcS3bdg/maxresdefault.jpg" alt="Tanıtım Videomuz" style="width:60%;"/>
  </a>
</div>

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

3. Anaconda ile sanal environment kurun (opsiyonel). Python kurulumu için 5. adıma atlayabilirsiniz :
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


# Web
Sosyal klasörü içerisinde yer alan dosyalarda çeşitli resimler, javascript fonksiyonları, HTML5 ve CSS kodları kullanılmıştır. Bu sayfa üzerinde profil oluşturma, aile üyeleri ve evcil hayvan bilgileri ekleme/düzenleme bölümleri bulunmaktadır. Aynı zamanda bu profiller sayesinde komşular tanınarak çeşitli özelliklere göre filtrelenebilecektir. Apartman yönetimi tarafından yapılacak etkinlikler ve olası toplantılar bu sayfada görünecek olup, alıp satılmak istenen eşyaların eklenmesi sağlanmaktadır.

Enerji klasörü içerisinde bulunan dosyalarda javascript fonksiyonları, HTML5 ve CSS kodları kullanılmıştır. Bu sayfada hanenin enerji tüketim grafiği, enerji tüketiminde fayda sağlayacak ve sorulara cevap veren sanal asistan, ortak kullanım alanlarının yoğunluk ve sıcaklıklarını kontrol etmeye ve görmeye yarayan bir ekran bulunmaktadır. Aynı zamanda günlük hava durumunu da yansıtan bir ekran yer almaktadır.

Kullanıcının etkileşime geçeceği uygulamaların tamamı matiricie.com/apsiyon klasörü içerisine de yüklenmiştir.

Canlı kullanım için sırasıyla; matiricie.com/apsiyon/sosyal ve matiricie.com/apsiyon/enerji adresleri ziyaret edilebilir.

# Mat-Man

Apsiyon’da olmayan özellikleri bünyesinde barındıran bir platform tasarlıyoruz. Platformumuz apartman yöneticiliğini kolaylaştırmayı ve inovatif hale getirmeyi amaçlamaktadır. 

Öncelikle platformumuz enerji verimliliğini artırmak için ortak alanlardaki klima, termostat, lamba gibi cihazlar hava durumu, içerideki insan yoğunluğu gibi faktörlere göre otomatik ayarlanacak ve apartman yöneticisine uzaktan müdahale hakkı sunacaktır. Aynı zamanda yine enerji verimliliği için apartman sakinlerinin hanelerinde kullandığı enerji miktarları takip edilecek ve anormal bir tüketim durumumda uyarı verecektir. Takip edilen enerji kullanım miktarları doğrultusunda yapay zeka ile kişiye özel tasarruf önerileri sunacaktır.

Komşuların birbirini tanıyamaz duruma geldiği apartmanlar için platformumuzda her bir daire bir profil oluşturarak komşularının kendilerini tanımasını sağlayabilecektir. İsteğe bağlı olarak girilecek çocuk sayısı ve yaşları, evcil hayvan, meslek gibi veriler filtreme özelliği sayesinde ihtiyaç durumundaki diğer komşulara yardımcı olacaktır. Daireler belirlenen tarihlere göre haftalık veya aylık olarak etkinlik önerisinde bulunabilecek ve bu öneri diğer sakinler tarafından anket aracılığıyla oylanarak belirlenebilecektir. Eşyalarını satmak isteyen sakinler platformumuzdaki eşya alım satım bölümüne yükleyerek apartman içerisinde alım veya satımını gerçekleştirebileceklerdir. 

Apartmanların bir diğer önemli sorunu olan park yeri ile ilgili platformumuzda bulunan otoparkın krokisinde park alanları dolu/boş olarak gösterilecektir. Aynı zamanda plaka okuma sistemi sayesinde daire sakinlerine kayıtlı bir alana farklı bir plaka park ettiğinde uyarı verebilecektir.
