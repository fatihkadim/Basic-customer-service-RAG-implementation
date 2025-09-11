"""
problem tanimi: akıllı müşteri destek sistemi
    -müsteriler sık sık benzer sorular sorar.

    -çözüm
        - .pdf dosyasını (sıkça sorulan sorular) vektör veritabanına dönüştürme(faiss)
        - kullanıcıdan gelen sorular veri tabanında sorgulanır ve gpt türkçe cevaplar üretir



Kullanılan teknolojiler
    -langchain: rag mimarisi kurmak için
    - faiss : embdedingleri saklmamak için hızlı bir vektör veri tabanı
    - openai: soru cevap için llm
    - streamlit: web arayüzü , son kullanıcı ile interaktif kullanıcı deneyimi


veriseti
    - Soru : yurtdısı satırslarınız bulunuyor mu
    - Cevap : hayır.

plan
    - SSS bilgilerini içeren pdf
    - kullanıcı dosyayı arayüzden yükleyecek
    - pdf metni parçalara ayrılacak ve embeddingler çıkarılacak
    - kullanıcı soru sorduğu zaman vektör db den benzer içerikler getirilir. gpt ile cevap oluşturulur.
    - konuşma geçmişi memory ile saklanır ve sonraki yanıtlara bağlam oluşturulur.


install libraries: freeze

"""