import React, { useState } from 'react';
import { Upload, Button, message, List, Spin, Card } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import $axios from '@/utils/axios'
import MyEcharts from '@/components/common/myEcharts'
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';

interface MyUploadProps {
  accept?: string;
}

function MyUpload({ accept }: MyUploadProps) {
  const [fileList, setFileList] = useState([]);
  const [uploadStatus, setUploadStatus] = useState<'success' | 'error' | null>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [downloadBtn, setDownloadBtn] = useState(false);
  const [chart, setChart] = useState(false);
  const [res, setRes] = useState([]);

  const getOption2 = () => {
    // 如果 res 的长度为 0，说明数据尚未准备好，返回一个空对象
    if (res.length === 0) {
      return {};
    }
    return {
      legend: {},
      tooltip: {},
      dataset: {
        source: [
          ['故障', '故障#0', '故障#1', '故障#2', '故障#3', '故障#4', '故障#5'],
          ['数量', res['typenum'][0], res['typenum'][1], res['typenum'][2],
            res['typenum'][3], res['typenum'][4], res['typenum'][5]],
        ]
      },
      xAxis: { type: 'category' },
      yAxis: {},
      series: [{ type: 'bar' }, { type: 'bar' }, { type: 'bar' }, { type: 'bar' }, { type: 'bar' }, { type: 'bar' }]
    }
  }

  const handleUpload = async (file: File) => {
    // 检查文件类型是否为 CSV
    if (file.type !== 'text/csv') {
      message.error('只能上传 CSV 文件！');
      return false; // 阻止默认的上传行为
    }

    try {
      setLoading(true); // 显示上传中的加载中动画
      setUploadStatus('success'); // 标记上传成功
      setLoading(false); // 关闭上传中的加载中动画
      setTraining(true); // 显示模型训练中的加载中动画

      const formData = new FormData();
      formData.append('file', file);

      await $axios
        .post('/predict/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        .then((res: any) => {
          res = JSON.parse(res)
          console.log(res);
          setRes(res);
          // getOption2();
          // console.log(JSON.parse(res));
          console.log(res['typenum']);

          message.success('样本预测完成！'); // 提示预测完成
          setTraining(false); // 关闭模型训练中的加载中动画
          setDownloadBtn(true);// 显示下载模型按钮
          setChart(true);// 显示分类图表
        })

    } catch (error) {
      console.error(error);
      message.error('出现故障！请刷新后重试');
      setUploadStatus('error'); // 标记上传失败
    } finally {
      setLoading(false); // 关闭上传中的加载中动画
      setTraining(false); // 关闭模型训练中的加载中动画
    }
  };

  const handleFileChange = ({ fileList }: any) => {
    setFileList(fileList);
    setUploadStatus(null); // 重置上传状态
    setDownloadBtn(false);
  };

  const downloadRes = () => {
    // 将数据转换为工作表对象
    const dataWithHeader = res['result1'].map((value: any, index: any) => ({ index, res: value }));
    const ws = XLSX.utils.json_to_sheet(dataWithHeader, { header: ['index', 'res'] });

    // 将工作表转换为工作簿对象
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');

    // 将工作簿保存为二进制数据
    const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'binary' });

    // 将二进制数据保存为文件
    const blob = new Blob([s2ab(wbout)], { type: 'application/octet-stream' });
    saveAs(blob, 'res.xlsx');
  };

  // 将字符串转换为 ArrayBuffer
  const s2ab = (s: string) => {
    const buf = new ArrayBuffer(s.length);
    const view = new Uint8Array(buf);

    for (let i = 0; i < s.length; i++) {
      view[i] = s.charCodeAt(i) & 0xff;
    }

    return buf;
  }

  return (
    <>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Upload
          fileList={fileList}
          beforeUpload={(file) => {
            handleUpload(file);
            return false; // 阻止默认的上传行为
          }}
          onChange={handleFileChange}
          showUploadList={false}
          accept={accept}
        >
          <Button icon={<UploadOutlined />}>上传 CSV 文件</Button>
        </Upload>
        <List
          dataSource={fileList}
          renderItem={(file) => (
            <List.Item>
              <List.Item.Meta title={file.name} />
              {uploadStatus === 'success' ? (
                <span style={{ color: 'green' }}>上传成功</span>
              ) : uploadStatus === 'error' ? (
                <span style={{ color: 'red' }}>上传失败</span>
              ) : null}
            </List.Item>
          )}
        />
        <div>{loading && <Spin size="large" tip="文件上传中，请稍候..." />}</div>
        <div>{training && <Spin size="large" tip="正在预测中，请稍候..." />}</div>
        <div>{downloadBtn && <Button onClick={downloadRes}>下载分类结果</Button>}</div>
        <br />
        <div>{chart && <div><Card>
          <MyEcharts
            option={getOption2()}
            style={{ width: '60vw', height: '500px' }}
          />
        </Card></div>}</div>
      </div>
    </>
  );
}

export default MyUpload;