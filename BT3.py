from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#20 câu mẫu
Cau = [
"Bài viết này sẽ đưa bạn đến những điểm du lịch tuyệt vời trên thế giới.",
"Các thông tin mới nhất về các sự kiện thể thao đang diễn ra trên toàn cầu.",
"Tìm hiểu về những địa điểm du lịch hấp dẫn và độc đáo.",
"Làm thế nào nghệ sĩ nổi tiếng đang ảnh hưởng đến thế giới văn nghệ.",
"Bí quyết để có một cuộc sống đầy đủ và hạnh phúc.",
"Thưởng thức những trải nghiệm du lịch không thể quên.",
"Cập nhật các thông tin mới nhất về các giải đấu thể thao quan trọng.",
"Làm thế nào để tận hưởng những khoảnh khắc tuyệt vời khi du lịch.",
"Phân tích sâu sắc về cuộc sống và đời sống hàng ngày.",
"Hướng dẫn du lịch cho những người đam mê khám phá.",
"Thảo luận về ảnh hưởng của nghệ thuật và văn hóa đối với cuộc sống.",
"Những điểm đến du lịch dành cho người yêu thể thao và phiêu lưu.",
"Phỏng vấn những nhân vật nổi tiếng trong lĩnh vực văn nghệ và giải trí.",
"Làm thế nào để tổ chức một chuyến du lịch hoàn hảo.",
"Đánh giá về cuộc sống hiện đại và những xu hướng mới.",
"Sự kiện thể thao lớn nào sẽ diễn ra trong thời gian tới?",
"Tìm hiểu về những người nổi tiếng và những hành động của họ trong lĩnh vực văn nghệ.",
"Hướng dẫn cách tận hưởng cuộc sống hằng ngày một cách tích cực.",
"Những món ăn ngon và độc đáo khi du lịch đến các địa điểm khác nhau.",
"Cuộc sống ở những thành phố sôi động và năng động.",
]
NhanY = ["DuLich",
         "Dulich",
         "TheThao",
         "DoiSong"]
#Khởi tạo CountVectorizer với các thiết lập mặc định
vec = CountVectorizer()
#Học từ điển và biểu diễn dữ liệu thành vector BoW
X = vec.fit_transform(Cau)

#Lấy DS các từ trong từ điển
voc = vec.get_feature_names_out()

# In DS từ vựng
print("DS Từ vựng: ",voc)

DataX = []
# In biểu diễn vector BoW cho mỗi câu
for i, Cau in enumerate(Cau):
    print(f"Biểu diễn vector BoW cho Câu {i + 1}:")
    print(X[i].toarray())    
    for each in X[i].toarray():
        DataX.append(each)
        print(DataX)

x_Train, x_Test, y_Train, y_Test = train_test_split(DataX, NhanY, test_size=0.25, random_state=30, shuffle=True )

print(len(x_Train))
print(len(x_Test))        
'''
from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(DataX, NhanY, test_size=0.2, random_state=42, shuffle=True)
print(trainX)
print(testX)
print(trainY)
print(testY)
'''