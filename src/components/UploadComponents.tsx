import React from "react";
import { InboxOutlined } from "@ant-design/icons";
import { Upload, message } from "antd";
import type { UploadProps } from "antd";
const { Dragger } = Upload;

const props: UploadProps = {
  name: "files",
  multiple: true,
  action: "http://127.0.0.1:8000/upload-multiple-files",
  accept: ".csv",
  beforeUpload(file, fileList) {
    console.log(file)
    if (fileList.length !== 2) {
      message.error("You must select at least 2 files.");
      return Upload.LIST_IGNORE;
    }
    return true;
  },
  onChange(info) {
    const { status } = info.file;
    if (status !== "uploading") {
      console.log(info.file, info.fileList);
    }
    if (status === "done") {
      message.success(`${info.file.name} file uploaded successfully.`);
    } else if (status === "error") {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
  onDrop(e) {
    console.log("Dropped files", e.dataTransfer.files);
  },
};

const UploadComponents: React.FC = () => (
  <div
    className="upload-container"
    style={{ padding: '30px' }}
  >
    <Dragger
      {...props}
      style={{
        fontSize: '30px',
        backgroundColor: "transparent",
        border: "2px dashed black",
        borderRadius: "8px",
        padding: "20px",

      }}
    >
      <p className="ant-upload-drag-icon">
        <InboxOutlined style={{ color: "black", fontSize: "48px" }} />
      </p>
      <p className="ant-upload-text">
        Click or drag file to this area to upload
      </p>
      <p className="ant-upload-hint">
        You must select at least 2 csv files to proceed.
      </p>
    </Dragger>
  </div>
);

export default UploadComponents;