/*
Basic gesture control using OpenCV.
(created by Kohtaro Tanaka(Student No. 03200271))

機能:
カメラに顔を近づけると写真が拡大され、顔をカメラから離すと写真が縮小される。拡大されているときに顔を右に４５度傾けると拡大される領域が右に動き、領域が画像の右端まで動いたら左下に領域が映る。顔を左に４５度傾けることで逆の操作を行うことができる。

どういうときに役に立つのか:
例えばレシピをタブレットやパソコンで見ながら料理をしたいとき、手でキーボードやマウスを触ることなく拡大縮小などの操作ができると非常に嬉しい。

流れ:
(1)まず、キャリブレーションを行う。拡大されるためにはどれほど顔を近づけるか、縮小されるためにはどれほど話せばよいかをユーザーに定義してもらう。
(2)もしズームインしたい場合は顔をカメラに近づけて、ズームアウトしたい場合はカメラから顔を話せば良い。カメラに顔を映らないようにしたり、中間の距離に顔を於けば画面は変化しない。

詳細の機能:
-ズームイン、ズームアウトの判断はint mainの中のr->widthを以下のように安定化処理(分散と平均を使って応答を安定化させる)したデータがキャリブレーションによって得られた閾値と比べて大きいか小さいかによって行う。
-比較に用いられるデータはより安定した応答がえられるように、連続する１０フレームのwidthの値の平均を使って比較し、分散がある一定の値（1000以下）でないと拡大・縮小の命令が出されないように設計した。
-顔を右に45度回転させた状態と左に４５度回転させた状態でも顔認識を行い、拡大領域の移動を可能にする。
-averageを計算したりするためのfor文はopenMPを使って高速化してある。
  

注：コメントはmain関数内を一番丁寧に書きました。その他の関数はすべてmain関数の上にかかれています。
 */



#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <omp.h>


 //
 //平均を計算
int average(int* n){
  int i;
  int s=0;
  #pragma omp parallel for
  for(i=0;i<10;i++){
#pragma omp critical
    s=s+n[i];
    //printf("s:%d\n",s);
  }
  return s/10;
 }
//分散計算のため平均の二乗を計算
int average_squared(int* n){
  int i;
  int s=0;
#pragma omp parallel for
  for(i=0;i<10;i++){
#pragma omp critical
    s=s+n[i]*n[i];
  }
  return s/10;
}
//分散を計算
int variance(int* n){
  return average_squared(n)-average(n)*average(n);
}


//画像を生成する関数に付随する
void display(cv::Mat image) {
  //出力画面に名前をつける
  std::string windowName = "window";
  cv::namedWindow(windowName);
  //画面出力
  cv::imshow(windowName, image);
  
  
}

//画像を生成する関数
void display_image(int width, int n){
  int movable_cols = 0;
  int movable_rows = 0;
  int move_cols = 0;
  int move_rows = 0;
  cv::Mat convert_mat, work_mat;
  //前処理　余った余白は黒で埋める
  work_mat = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);
  //画面読み込み
  cv::Mat target_mat= cv::imread("presentation.png");
  
  //縦横どっちか長い方は？
  int big_width = target_mat.cols > target_mat.rows ? target_mat.cols : target_mat.rows;
  //割合
  double ratio = ((double)width / (double)big_width);
  
  //リサイズ
  cv::resize(target_mat, convert_mat, cv::Size(), ratio, ratio, cv::INTER_NEAREST);

  //中心をアンカーにして配置
  if(convert_mat.cols<1001 && convert_mat.rows < 1001){
    cv::Mat Roi1(work_mat, cv::Rect((1000 - convert_mat.cols) / 2, (1000 - convert_mat.rows) / 2, convert_mat.cols, convert_mat.rows));
    convert_mat.copyTo(Roi1);
  }
  else{
    //クロップ。クロップ範囲はmain関数によって定義。顔を横に傾けるとクロップする範囲を移動させることができる。
    movable_cols = convert_mat.cols-1000; //1000*1000の枠が左右に何回動くことができるか？
    movable_rows = convert_mat.rows-1000;//1000*1000の枠が上下に何回動くことができるか？
    move_cols = n % movable_cols;//実際に横に動かす
    move_rows = n / movable_cols * 80;//実際に縦に動かす

    if(move_rows < (movable_rows+1)){
      cv::Rect rect = cv::Rect(move_cols,move_rows,1000,1000);
      cv::Mat roi_img(convert_mat, rect);
      cv::Mat Roi1(work_mat, cv::Rect((1000 - roi_img.cols) / 2, (1000 - roi_img.rows) / 2, roi_img.cols, roi_img.rows));
      roi_img.copyTo(Roi1);
    }
    else{
      cv::Rect rect = cv::Rect(move_cols,movable_rows,1000,1000);
      cv::Mat roi_img(convert_mat, rect);
      cv::Mat Roi1(work_mat, cv::Rect((1000 - roi_img.cols) / 2, (1000 - roi_img.rows) / 2, roi_img.cols, roi_img.rows));
      roi_img.copyTo(Roi1);
    }
    
  }
  
  target_mat = work_mat.clone();
  
  //画面へ
  display(target_mat);
}





 //calibrationステップ１の開始案内表示
void zoom_in_calibration_start(void){

  cv::Mat img = cv::Mat::zeros(600,900,CV_8UC3);
  
  cv::putText(img, "Calibration Step 1", cv::Point(270,220), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(200,200,100), 2, CV_AA);
  cv::putText(img, "Place your face close to the camera.", cv::Point(192,280), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(200,200,100), 2, CV_AA);
  cv::putText(img, "Press 'Esc' or wait 15 seconds to the continue.", cv::Point(150,340), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(200,200,100), 2, CV_AA);
  cv::putText(img, "Calibration may take up to 30 seconds", cv::Point(190,400), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(200,200,100), 2, CV_AA);
  
  cv::namedWindow("Calibration", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("Calibration", img);
  cv::waitKey(1000*16);
  cv::destroyWindow ("Calibration");
  
}

//calibration作業(ステップ１，２共通)
int calibration(void){

  int ave=0;
  int counter = 0;
  int data[120]; //calibrationデータ格納用
  cv::Mat frame, frame_img_gray;
  const char *cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml";
  
  cv::CascadeClassifier cascade;
  std::vector<cv::Rect> faces;
  static cv::Scalar colors[] = {
    {0, 0, 255}, {0, 128, 255},
    {0, 255, 255}, {0, 255, 0},
    {255, 128, 0}, {255, 255, 0},
    {255, 0, 0}, {255, 0, 255}
  };

  unsigned char c;
  
  // 指定された番号のカメラに対するキャプチャオブジェクトを作成する
  cv::VideoCapture capture(0);
  
  // 表示用ウィンドウをの初期化
  cv::namedWindow ("Calibration", CV_WINDOW_AUTOSIZE);
  
  while (capture.isOpened()) {
    
    // カメラから画像をキャプチャする
    capture.read(frame);
    
    // ブーストされた分類器のカスケードを読み込む
    cascade.load(cascade_name);

    // 読み込んだ画像のグレースケール化，ヒストグラムの均一化を行う
    cv::cvtColor (frame, frame_img_gray, CV_BGR2GRAY);
    cv::equalizeHist (frame_img_gray, frame_img_gray);

    //顔の検出を行う。
    cascade.detectMultiScale(frame_img_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(40, 40) );
    
    //検出された顔位置の端の２点に，円を描画する。プログラムの確認用。
    int i = 0;
    int wid = 0;
    int wid_ave;
    int var;
    for(std::vector<cv::Rect>::iterator r = faces.begin(); r != faces.end(); r++){
      cv::Point center1, center2;
      int radius;
      r=r;
      center1.x = cvRound(r->x);
      center1.y = cvRound(r->y);
      center2.x = cvRound(r->x+r->width);
      center2.y = cvRound(r->y);
      radius = 20;
      cv::Scalar dot(0, 255, 255);
      int draw_in = -1;
      cv::circle(frame, center1, radius, dot, draw_in, cv::LINE_AA);
      cv::circle(frame, center2, radius, dot, draw_in, cv::LINE_AA);
      wid = cvRound(r->width);
      data[counter]=wid;
      if(counter!=119){
	counter=counter+1;
      }
      else if(counter==119){
	int s=0;
#pragma omp parallel for
	for(i=20;i<120;i++){
#pragma omp critical
	  s=s+data[i];
	}
	ave=s/100;
	printf("calibration:%d\n",ave);
	return ave;
      }
      
      else{
	break;
	printf("Error!!\nProcess terminated\n");
      }
      
    }

    //calibration in progressの文字表示
    cv::putText(frame, "CALIBRATION IN PROGRESS", cv::Point(70,50), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(10,10,200), 2, CV_AA);
    
    // (4) カメラ画像の表示
    cv::imshow ("Calibration", frame);
    

    // (5) 1msecだけキー入力を待つ
    c = cv::waitKey (1);
    if (c == '\x1b') // Escキー
      break;
    
  }
  
  cv::destroyWindow ("Calibration");
  return ave;
}

 //calibrationステップ１の終了案内表示
void zoom_in_calibration_end(void){
  
  cv::Mat img = cv::Mat::zeros(600,900,CV_8UC3);
  
  cv::putText(img, "Calibration Step 1 Completed", cv::Point(170,220), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(200,200,100), 2, CV_AA);

  cv::namedWindow("Calibration", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("Calibration", img);
  cv::waitKey(1000*3);
  cv::destroyWindow ("Calibration");
  
}



 //calibrationステップ２の開始案内表示
void zoom_out_calibration_start(void){

  cv::Mat img = cv::Mat::zeros(600,900,CV_8UC3);
  
  cv::putText(img, "Calibration Step 2", cv::Point(270,220), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(200,200,100), 2, CV_AA);
  cv::putText(img, "Place your face away from the camera.", cv::Point(192,280), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(200,200,100), 2, CV_AA);
  cv::putText(img, "Press 'Esc' or wait 15 seconds to the continue.", cv::Point(150,340), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(200,200,100), 2, CV_AA);
  cv::putText(img, "Calibration may take up to 30 seconds", cv::Point(190,400), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(200,200,100), 2, CV_AA);
  
  cv::namedWindow("Calibration", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("Calibration", img);
  cv::waitKey(1000*16);
  cv::destroyWindow ("Calibration");
  
}

 //calibration zooステップ２の終了案内表示
void zoom_out_calibration_end(void){
  //calibration zoominの案内表示
  cv::Mat img = cv::Mat::zeros(600,900,CV_8UC3);
  
  cv::putText(img, "Calibration Step 2 Completed", cv::Point(170,220), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(200,200,100), 2, CV_AA);

  cv::namedWindow("Calibration", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("Calibration", img);
  cv::waitKey(1000*3);
  cv::destroyWindow ("Calibration");
  
}




























//main関数




int main (int argc, char **argv){

  int zoom_in_calibrated;//calibrationステップ１によって得られた閾値格納用
  int zoom_out_calibrated;//calibrationステップ２によって得られた閾値格納用
  
  //(1)calibrationステップ１の開始案内
  zoom_in_calibration_start();
  //(2)calibrationステップ１の実行
  zoom_in_calibrated=calibration();
  //(3)calibrationステップ１の終了案内
  zoom_in_calibration_end();
  
  //(4)calibrationステップ２の開始案内
  zoom_out_calibration_start();
  //(5)calibrationステップ２の実行
  zoom_out_calibrated=calibration();
  //(6)calibrationステップ２の終了案内
  zoom_out_calibration_end();
 

  //以下顔認識とジェスチャーコントロールのプログラム

  int n = 0;//右方向へのスクロール数（単位ピクセル）（注：一段下に動いてもｎの数値は連続している）
  int size = 1000;//初期イメージサイズ
  
  //(7)初期画像を表示
  display_image(size,n);

  
  cv::Mat frame, frame_img_gray;
  const char *cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml";
  
  cv::CascadeClassifier cascade;
  std::vector<cv::Rect> faces;
  static cv::Scalar colors[] = {
    {0, 0, 255}, {0, 128, 255},
    {0, 255, 255}, {0, 255, 0},
    {255, 128, 0}, {255, 255, 0},
    {255, 0, 0}, {255, 0, 255}
  };

  unsigned char c;
  
  // (8)指定された番号のカメラに対するキャプチャオブジェクトを作成する
  cv::VideoCapture capture(0);
  
  // (9)表示用ウィンドウの初期化
  cv::namedWindow ("face recognition", CV_WINDOW_AUTOSIZE);

  int counter = 0;//顔の端点距離データ取得回数格納用
  int data[10]; //顔の端点距離データ格納用
  while (capture.isOpened()) {
    
    // (10.1)カメラから画像をキャプチャする
    capture.read(frame);
    
    // (10.2)ブーストされた分類器のカスケードを読み込む
    cascade.load(cascade_name);

    // (10.3)読み込んだ画像のグレースケール化，ヒストグラムの均一化を行う
    cv::cvtColor (frame, frame_img_gray, CV_BGR2GRAY);
    cv::equalizeHist (frame_img_gray, frame_img_gray);
    
    
    
    // (10.4)顔の検出
    cascade.detectMultiScale(frame_img_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(40, 40) );
    
    // (10.5)検出された全ての顔位置の左右の端点に，円を描画する
    int i = 0;

    for(std::vector<cv::Rect>::iterator r = faces.begin(); r != faces.end(); r++){
      cv::Point center1, center2;
      int radius;
      r=r;
      center1.x = cvRound(r->x);
      center1.y = cvRound(r->y);
      center2.x = cvRound(r->x+r->width);
      center2.y = cvRound(r->y);
      radius = 20;
      cv::Scalar dot(0, 255, 255);
      int draw_in = -1;
      cv::circle(frame, center1, radius, dot, draw_in, cv::LINE_AA);
      cv::circle(frame, center2, radius, dot, draw_in, cv::LINE_AA);


      //(10.6)顔の左右２点間距離を取得、格納する
      int wid = 0;//顔の左右２端点間距離格納用
      int wid_ave;//顔の左右２端点間距離平均格納用
      int var;//顔の左右２端点間距離分散格納用
      wid = cvRound(r->width);
      data[counter]=wid;
      //printf("counter:%d\n",counter);

      //(10.7-1)もし１０回データを取得していないならカウンタを上げる
      if(counter!=9){
	counter=counter+1;
	//data[counter]=wid;
      }

      //(10.7-2)10回データを取得したら平均、分散を計算し、分散が１０００以下かつ平均がステップ１の閾値-20より大きいならズームイン(画像sizeを増やす)、ステップ２の閾値+20より小さいならズームアウト(画像サイズを減らす)
      else if(counter==9){
	//data[counter]=wid;
	counter=0;
	wid_ave=average(data);
	var=variance(data);
	printf("variance:%d\n",var);
	//if average width is greater than zoom_in_calibrated-20かつ分散が1000以下である場合はzoom inする。
	if(wid_ave>zoom_in_calibrated-20 && var<1000){
	  cv::putText(frame, "ZOOM IN", center1, cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,200), 3, CV_AA);
	  size = size + 40;
	}
	//if average width is less than zoom_in_calibrated-20かつ分散が1000以下である場合はzoom outする。
	else if(wid_ave<zoom_out_calibrated+20 && var<1000){
	  cv::putText(frame, "ZOOM OUT", center1, cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,200), 3, CV_AA);
	  if(size>50){
	    size = size - 40;
	  }
	}
	else{
	  cv::putText(frame, "DO NOTHING", center1, cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,200), 3, CV_AA);
	}
      }
      //エラー処理
      else{
	break;
	printf("Error!!\nProcess terminated\n");
      }
    }
    
    
    
    //(10.8)画像の-45度回転・顔認識
    
    cv::Mat src_img = frame_img_gray;
     
    
    // 回転： -45 [deg],  スケーリング： 1.0 [倍]
    float angle = -45.0, scale = 1.0;
    // 中心：画像中心
    cv::Point2f center(src_img.cols*0.5, src_img.rows*0.5);
    // 以上の条件から2次元の回転行列を計算
    const cv::Mat affine_matrix = cv::getRotationMatrix2D( center, angle, scale );
    
    cv::Mat dst_img;
    cv::warpAffine(src_img, dst_img, affine_matrix, src_img.size());
    
    //(10.9)顔を検出したら右にスクロール(nの値を増やす)
    cascade.detectMultiScale(dst_img, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(40, 40) );
    for(std::vector<cv::Rect>::iterator r = faces.begin(); r != faces.end(); r++){
      n = n + 20;
    }
    
    
    //(10.10)画像の45度回転・顔認識
    cv::Mat src_img2 = frame_img_gray;

    
    // 回転： 45 [deg],  スケーリング： 1.0 [倍]
    angle = 45.0, scale = 1.0;
    // 中心：画像中心
    cv::Point2f center2(src_img2.cols*0.5, src_img2.rows*0.5);
    // 以上の条件から2次元の回転行列を計算
    const cv::Mat affine_matrix2 = cv::getRotationMatrix2D( center2, angle, scale );
    
    cv::Mat dst_img2;
    cv::warpAffine(src_img2, dst_img2, affine_matrix2, src_img2.size());

    // (10.11)顔を検出したら左にスクロール（ｎが負にならないようにｎの値を減らす）
    cascade.detectMultiScale(dst_img2, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(40, 40) );
    for(std::vector<cv::Rect>::iterator r = faces.begin(); r != faces.end(); r++){
      if(n-1>-1){
	n = n - 20;
      }
    }
    
    //(10.12)新たなサイズ・位置の画像を表示
    display_image(size,n);
    
    //(10.13) カメラ画像の表示。確認用
    cv::imshow ("face recognition", frame);
    
    //(10.14) 1msecだけキー入力を待つ
    c = cv::waitKey (1);
    if (c == '\x1b') // Escキー
      break;
  }

  //(11)windowを削除
  cv::destroyWindow ("face recognition");
  
  return 0;
}

