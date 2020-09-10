<template>
  <div class="container">
    <h2>Video Object Detection</h2>
    <div class="row">
      <label> File </label>
      <input
        class="form-control-file"
        type="file"
        id="file"
        ref="file"
        v-on:change="handleFileUpload()"
      />
      <button type="button" class="btn btn-primary" v-on:click="submitFile()">
        Submit
      </button>
      <button type="button" class="btn btn-danger" v-on:click="[detectFile(), process()]">
        Dectect
      </button>
            <button type="button" class="btn btn-success" v-on:click="run()">
        Run
      </button>
    </div>
    <div class="row">
      <h5>{{message}}</h5>
    </div>
    <div v-if="this.status == 'DetectDone'" class="container two">
    <!-- <div class="container two"> -->
      <video width="960" height="540" controls>
        <!-- <source :src= "path" type="video/mp4" /> -->
        <source src= "@/assets/upload.mp4" type="video/mp4" />
        <!-- <source src= "./../assets/upload.mp4" type="video/mp4" /> -->
      </video>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "Detection",
  data() {
    return {
      file: "",
      message:"Ready..",
      status : "not",
      path:""
    };
  },
  methods: {
    handleFileUpload() {
      this.file = this.$refs.file.files[0];
      this.status = 'Loading'
      this.message = "Nothing here"
      console.log(this.file);
      console.log("oki");
    },
    process(){
      this.message = "Processing data..."
    },
    run(){
      this.status = 'DetectDone'
    },
    detectFile() {
      axios
      .get("http://127.0.0.1:5000/detect")
      .then(async (res) =>{
        console.log(res.data);
        // this.path = require("../assets/upload.mp4")
        // this.path = "/media/upload.fe9d825f.mp4"
        
        this.message = "Detection Done";
        this.status = 'DetectDone';  
        await location.reload();
      })
      
    },
    submitFile() {
      let formData = new FormData();
      formData.append("file", this.file);
      console.log(formData);
      axios
        .post("http://127.0.0.1:5000/upload-file", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        })
        .then((res) => {
          console.log(res.data)
          this.message = "Upload Done"
          console.log("SUCCESS!!");
        })
        .catch(function (err) {
          console.log(err);
          console.log("FAILURE!!");
        });
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 20px 0;
}
label,
#file,
.btn {
  display: inline-block;
  width: auto;
  vertical-align: middle;
  margin-right: 30px;
}
label {
  margin-left: 300px;
  margin-right: 30px;
}
.row {
  margin-top: 10px;
  vertical-align: middle;
}
.two {
  margin-top: 20px;
}
.video {
  border-color: red;
  border: 2px;
}
h5{
  width: 100%;
  color: red;
}
</style>
