import React, { useState } from 'react';
import { Upload, Button, message, List, Spin } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import Papa from 'papaparse'; // 导入 PapaParse 库
import $axios from '@/utils/axios'

interface MyUploadProps {
  accept?: string;
}

function MyUpload({ accept }: MyUploadProps) {
  const [fileList, setFileList] = useState([]);
  const [uploadStatus, setUploadStatus] = useState<'success' | 'error' | null>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState('');
  const [plotUrl, setPlotUrl] = useState(null);

  const handleUpload = async (file: File) => {
    // 检查文件类型是否为 CSV
    if (file.type !== 'text/csv') {
      message.error('只能上传 CSV 文件！');
      return false; // 阻止默认的上传行为
    }

    try {
      setLoading(true); // 显示上传中的加载中动画
      const { data } = await new Promise<Papa.ParseResult>((resolve, reject) => {
        Papa.parse(file, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            resolve(results);
          },
          error: (error) => {
            reject(error);
          },
        });
      });
      // console.log(data); // 将解析结果打印到控制台
      setUploadStatus('success'); // 标记上传成功
      setLoading(false); // 关闭上传中的加载中动画
      setTraining(true); // 显示模型训练中的加载中动画

      await $axios
        .post('/predict/', data)
        .then((res: any) => {
          // console.log(res);
          setTrainResult(''); // 保存训练结果
          message.success('模型训练完成！'); // 提示训练完成
          setTraining(false); // 关闭模型训练中的加载中动画
          // TODO: 接收后端传来的图像并显示 base64
          // 将 PNG 图像数据转换为 base64 编码的字符串
          const binaryData = atob(res);
          const blob = new Blob([new Uint8Array(binaryData.length).map((_, i) => binaryData.charCodeAt(i))], { type: 'image/png' });
          const imageUrl = URL.createObjectURL(blob);
          setPlotUrl(imageUrl);
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
    setTrainResult(''); // 重置训练结果
  };

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
        <div>{training && <Spin size="large" tip="模型正在训练中，请稍候..." />}</div>
        <div>{trainResult && <div>训练结果：{trainResult}</div>}</div>
        <div>{plotUrl && <img src={plotUrl} alt="Training Plot" />}</div>
      </div>
    </>
  );
}

export default MyUpload;