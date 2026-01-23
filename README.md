# Adil Backend

Adil, kullanıcıların günlük hayatta karşılaştıkları durumlarda temel hukuki haklarını öğrenebilmeleri için tasarlanmış bir mobil uygulamadır. Bu depo, Adil uygulamasında kullanılan chatbotun backend servislerini içerir. Kullanıcıdan gelen soruları alır, gerekli işlemleri yapar ve cevapları JSON formatında mobil uygulamaya döner.

Başlangıçta geliştirme ve deneme amaçlı hazırlanmıştır; ilerleyen aşamalarda üretim ortamına uygun güvenlik, loglama ve ölçekleme adımları eklenecektir.


## Teknik Özellikler

- Python tabanlı backend (FastAPI)
- REST API ile mobil uygulamaya cevap dönen chatbot endpoint’i
- Geliştirme için lokal, yayın için HTTPS destekli deploy planı
