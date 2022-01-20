# Barcode_Detector
1. Mục tiêu.
- Xác định được vị trí mã vạch qua hình ảnh và camera:
  + Hình ảnh: xác định được vị trí mã vạch khi ảnh xoay chưa đúng vị trí hay trong vùng có nhiều chữ.
  + Camera: xác định được vị trí mã vạch trên không gian thực (ghi hình trực tiếp).
- Xây dựng một ứng dụng nhận dạng mã vạch.

2. Thực hiện thông qua ba bước xử lý với hình ảnh:
- Xác định vùng nhận diện: Sử dụng bộ lọc Sobel cả hai chiều ngang và dọc để nhận dạng các cạnh, từ đó xác định vùng cần nhận diện mã vạch. Sobel có khả năng chống nhiễu tốt.
- Xoay hình ảnh về đúng vị trí.
- Nhận dạng các cạnh với phương pháp tích chập filter2D kết hợp với phương pháp nhận dạng đường thẳng Probabilistic Hough Line Transform.

3. Nhận diện mã vạch: 
Về cơ bản cũng thực hiện phương pháp xử lý giống với bước Xoay ảnh về đúng vị trí nhưng kết hợp thêm với Morphological Shape.

4. Đối với camera: 
Sử dụng bộ lọc Sobel cả hai chiều ngang và dọc với Scharr kernel và kết hợp với Morphological Shape. 

Tài liệu tham khảo: <br/>
[1] Image Gradients with OpenCV (Sobel and Scharr) – PyImageSearch: https://www.pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/ <br/>
[2] Convolutions with OpenCV and Python – PyImageSearch: https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/ <br/>
[3] OpenCV Morphological Operations – PyImageSearch: https://www.pyimagesearch.com/2021/04/28/opencv-morphological-operations/ <br/>
