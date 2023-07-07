// import React from 'react'

// const Workspace = () => {
//   return (
//     <div>
//       <h1>WorkSpace</h1>
//     </div>
//   )
// }

// export default Workspace
import { InboxOutlined } from '@ant-design/icons';
// import type { UploadProps } from 'antd';
import { message, Upload, Spin, Alert } from 'antd';
import React from 'react';
import { useState } from 'react';
import style from './Home.module.less'
import Papa from 'papaparse'
import $axios from '@/utils/axios'

const { Dragger } = Upload;



const Workspace: React.FC = () => {
  const [status, setStatus] = useState('');
  const props: any = {
    name: 'file',
    multiple: true,
    accept: '.csv',
    action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
    onChange(info) {
      const { status, originFileObj } = info.file;
      if (status !== 'uploading') {
        console.log(info.file, info.fileList);
      }
      if (status === 'done') {
        message.success(`${info.file.name} file uploaded successfully.`);
        // 解析csv文件
        const reader = new FileReader();
        reader.onload = (e) => {
          const csvData = e.target.result;
          Papa.parse(csvData, {
            header: true,
            complete: (parsedData) => {
              // 拿到解析后的数据
              console.log(parsedData.data);
              // 把csv中数据传给后端predict接口
              // 显示loading
              setStatus('training');
              $axios
                .post('/predict/', parsedData.data)
                .then((res) => {
                  console.log(res);
                  // 结束loading
                  setTimeout(() => setStatus('complete'), 1000);
                  // 接收后端传来的图像并显示 base64
                })
                .catch(err => console.log(err))
            },
            error: (error) => {
              console.error('Error parsing CSV file:', error);
            },
          });
        };
        reader.readAsText(originFileObj);
      } else if (status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
  };

  return (
    <div className={style.cusDraggerBox}>
      <Dragger {...props}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">Click or drag file to this area to upload</p>
        <p className="ant-upload-hint">
          Support for a single or bulk upload. Only csv file is ok.
        </p>
      </Dragger>
      {/* <TrainingRes status={status}></TrainingRes> */}
      {status === 'training' ? <div className={style.cusWorkSpace}>训练需要一段时间，请耐心等待......&nbsp;<Spin></Spin></div> :
        (status === 'complete' ?
          <div className={style.cusWorkSpace}>
            <Alert
              message="训练完成"
              description="分类结果呈现如下"
              type="info"
            />
          </div>
          : <div />
        )}
    </div>
  )
};

export default Workspace;