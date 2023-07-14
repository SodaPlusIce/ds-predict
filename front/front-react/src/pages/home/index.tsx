import React, { useState } from 'react';
import { Upload, Button, message, List, Spin } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import $axios from '@/utils/axios'

interface MyUploadProps {
  accept?: string;
}

function MyUpload({ accept }: MyUploadProps) {
  const [fileList, setFileList] = useState([]);
  const [uploadStatus, setUploadStatus] = useState<'success' | 'error' | null>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [downloadBtn, setDownloadBtn] = useState(false);
  const [plotUrl, setPlotUrl] = useState(null);
  const [trainingTime, setTrainingTime] = useState<number | null>(null);

  const handleUpload = async (file: File) => {
    // 检查文件类型是否为 CSV
    if (file.type !== 'text/csv') {
      message.error('只能上传 CSV 文件！');
      return false; // 阻止默认的上传行为
    }

    try {
      setLoading(true); // 显示上传中的加载中动画
      const startTime = new Date().getTime(); // 记录模型训练开始的时间
      setUploadStatus('success'); // 标记上传成功
      setLoading(false); // 关闭上传中的加载中动画
      setTraining(true); // 显示模型训练中的加载中动画

      const formData = new FormData();
      formData.append('file', file);

      await $axios
        .post('/train/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        .then((res: any) => {
          console.log(res);
          message.success('模型训练完成！'); // 提示训练完成
          const endTime = new Date().getTime(); // 记录模型训练结束的时间
          const timeDiff = Math.round((endTime - startTime) / 1000); // 计算训练用时（单位为秒）
          setTrainingTime(timeDiff); // 记录训练用时
          setTraining(false); // 关闭模型训练中的加载中动画
          setDownloadBtn(true);// 显示下载模型按钮
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

  const downloadModel = async () => {
    try {
      const response:any = await $axios.get('/download/', { responseType: 'blob' });
      
      const url = window.URL.createObjectURL(new Blob([response]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'model.bin');
      document.body.appendChild(link);
      link.click();
    } catch (error) {
      console.error(error);
    }
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
                <span style={{ color: 'red' }}>上传失败</span>)
                : null}
            </List.Item>
          )}
        />
        <div>{loading && <Spin size="large" tip="文件上传中，请稍候..." />}</div>
        <div>{training && <Spin size="large" tip="模型正在训练中，请稍候..." />}</div>
        <div>{downloadBtn && <Button onClick={downloadModel}>下载模型</Button>}</div>
        {trainingTime && <div>训练用时：{trainingTime} 秒</div>}
        {/* <div>{plotUrl && <img src={plotUrl} alt="Training Plot" />}</div> */}
        <br />
      </div>
    </>
  );
}

export default MyUpload;