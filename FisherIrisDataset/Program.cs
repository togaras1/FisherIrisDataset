using System;
using System.Linq;
using System.Diagnostics;
using System.Collections.Generic;
using System.IO;

namespace FisherIrisDataset
{
    struct iris{
        public iris(double sl, double sw, double pl, double pw,string s){
            Species = s;
            SepalLength = sl;
            SepalWidth = sw;
            PetalLength = pl;
            PetalWidth = pw;
        }

        public string Species { get; set; }
        public double SepalLength { get; set; }
        public double SepalWidth { get; set; }
        public double PetalLength { get; set; }
        public double PetalWidth { get; set; }
    }

    class MainClass
    {
        static List<iris> dataset = new List<iris>();

        public static void Main(string[] args)
        {
            // データ読み込み
            var lines = File.ReadAllLines("iris.data");
            lines.Select(m => {
                var n = m.Split(',');
                dataset.Add(new iris(double.Parse(n[0]),double.Parse(n[1]),double.Parse(n[2]),double.Parse(n[3]),n[4]));
                return m;
            }).ToArray();

            // 名前リスト(重複なし)
            var labelName = dataset.Select(m => m.Species).Distinct().ToArray();

            // 特徴ベクトル
            var feature = (from n in dataset select new double[] { n.PetalLength, n.PetalWidth, n.SepalLength, n.SepalWidth }).ToArray();
            // 名前リストを数字リストに変換する
            var labelNumber = dataset.Select(m => Array.IndexOf(labelName, m.Species)).ToArray();

            /*
            int num = 1;
            
            // 訓練データ
            var train = feature.Select((x, i) => new { Content = x, Index = i })
                               .Where(x => x.Index % 50 < 50-num)
                               .Select(x => x.Content)
                               .ToArray();
            // 訓練ラベル
            var trainl = labelNumber.Select((x, i) => new { Content = x, Index = i })
                                    .Where(x => x.Index % 50 < 50-num)
                                    .Select(x => x.Content)
                                    .ToArray();
            // テストデータ
            var test = feature.Select((x, i) => new { Content = x, Index = i })
                              .Where(x => x.Index % 50 >= 50-num)
                              .Select(x => x.Content)
                              .ToArray();
            // テストデータラベル
            var testl = labelNumber.Select((x, i) => new { Content = x, Index = i })
                                   .Where(x => x.Index % 50 >= 50-num)
                                   .Select(x => x.Content)
                                   .ToArray();
            */

            for (int num = 0; num < 50; ++num)
            {
                // 訓練データ
                var train = feature.Select((x, i) => new { Content = x, Index = i })
                                   .Where(x => x.Index % 50 != num && x.Index < 100)
                                   .Select(x => x.Content)
                                   .ToArray();
                // 訓練ラベル
                var trainl = labelNumber.Select((x, i) => new { Content = x, Index = i })
                                        .Where(x => x.Index % 50 != num && x.Index < 100)
                                        .Select(x => x.Content)
                                        .ToArray();
                var test = feature.Select((x, i) => new { Content = x, Index = i })
                                  .Where(x => x.Index % 50 == num && x.Index < 100)
                                  .Select(x => x.Content)
                                  .ToArray();
                var testl = labelNumber.Select((x, i) => new { Content = x, Index = i })
                                       .Where(x => x.Index % 50 == num && x.Index < 100)
                                       .Select(x => x.Content)
                                       .ToArray();

                var svm = new SVM(train, trainl);
                svm.Learn();
                var result = test.Select(x => svm.Predict(x)).ToArray();
                result.Select((m, i) => { Console.WriteLine("Number{0}: Result:{1} Answer:{2}", num, labelName[m], labelName[testl[i]]); return m; }).ToArray();
            }
        }
    }
}