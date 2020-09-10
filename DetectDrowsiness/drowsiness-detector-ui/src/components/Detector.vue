<template>
  <div id='detector' class="detector">
    <h1 class='instagram'>Driver Drowsiness Detector</h1>
    <center>
      by HaPhamD
    </center>
    <br>
    <center>
        <div class='frame'>
            <img src='http://localhost:8000/face_detector' alt='webcam'>
        </div>
    </center>
    <center>
      <div v-if="this.status == 'done'">
          <span class='gradient-fill'>Result</span>
          <br>
          <span>{{ result }}</span>
      </div>
    </center>
  </div>
</template>

<script>

import axios from 'axios';

export default {
  name: 'detector',
  data() {
    return {
      status: 'started',
      result: 'default',
    };
  },
  mounted() {
    this.status = 'loading';
    axios
      .get('http://localhost:8000/get_result')
      .then((response) => {
        this.result = response.data.result;
        this.status = 'done';
      })
      .catch((error) => {
        console.error(error);
      });
  },
};
</script>

<style scoped>
.detector {
  text-align: left;
  font-family: Inter, Inter UI, Inter-UI, SF Pro Display, SF UI Text,
    Helvetica Neue, Helvetica, Arial, sans-serif;
  font-weight: 400;
  letter-spacing: +0.37px;
  color: rgb(175, 175, 175);
}

img {
    border-radius: 5%;
}

.frame {
  width: 420px;
  height: 305px;
  border: 5px solid #ae41a7;
  border-radius: 5%;
  background: #eee;
  margin: center;
  padding: 15px 10px;
}

.form-control:focus {
  border-color: #ae41a7 !important;
  box-shadow: 0 0 5px #ae41a7 !important;
}

.instagram{
  text-align: center;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-weight: 800;
  letter-spacing: +0.37px;
  color: rgb(255, 255, 255);
  background: #f09433;
  background: -moz-linear-gradient(
    45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #bc1888 100%
  );
  background: -webkit-linear-gradient(
    45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #806878 100%
  );
  background: linear-gradient(45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #bc1888 100%
  );
  filter: progid:DXImageTransform.Microsoft.gradient( startColorstr=#f09433,
    endColorstr=#bc1888,
    GradientType=1
  );
}

.header {
  text-align: center;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-weight: 800;
  letter-spacing: +0.37px;
  color: rgb(255, 255, 255);
  background-image: linear-gradient(
    -225deg,
    #a445b2 0%,
    #d41872 52%,
    #ff0066 100%
  );
}

.gradient-fill {
  background-image: linear-gradient(
    -225deg,
    #a445b2 0%,
    #d41872 52%,
    #ff0066 100%
  );
}

.gradient-fill.background {
  background-size: 250% auto;
  border: medium none currentcolor;
  border-image: none 100% 1 0 stretch;
  transition-delay: 0s, 0s, 0s, 0s, 0s, 0s;
  transition-duration: 0.5s, 0.2s, 0.2s, 0.2s, 0.2s, 0.2s;
  transition-property: background-position, transform, box-shadow, border,
    transform, box-shadow;
  transition-timing-function: ease-out, ease, ease, ease, ease, ease;
  color: white;
  font-weight: bold;
  border-radius: 3px;
}

span.gradient-fill {
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 20px;
  font-weight: 700;
  line-height: 2.5;
}

</style>
