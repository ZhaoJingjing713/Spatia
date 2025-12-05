<div align="center">

<h1 align="center">
  <span style="color: #4F46E5; font-size: 1.2em;">Spatia</span>: Video Generation with Updatable Spatial Memory
</h1>

<p align="center" style="font-size: 1.1em; color: #555;">
  <strong>Long-horizon, spatially consistent video generation enabled by persistent 3D scene point clouds and dynamic-static disentanglement.</strong>
</p>

<div align="center">
  <a href="https://github.com/ZhaoJingjing713">Jinjing Zhao</a><sup>*1</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=-ncz2s8AAAAJ">Fangyun Wei</a><sup>*2</sup>&nbsp;&nbsp;
  <a href="https://www.liuzhening.top">Zhening Liu</a><sup>3</sup>&nbsp;&nbsp;
  <a href="https://hongyanz.github.io/">Hongyang Zhang</a><sup>4</sup>&nbsp;&nbsp;
  <a href="http://changxu.xyz/">Chang Xu</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=djk5l-4AAAAJ">Yan Lu</a><sup>2</sup>
</div>

<p align="center" style="font-size: 0.9em; color: #666;">
  <sup>1</sup>The University of Sydney&nbsp;&nbsp;
  <sup>2</sup>Microsoft Research Asia&nbsp;&nbsp;
  <sup>3</sup>HKUST&nbsp;&nbsp;
  <sup>4</sup>University of Waterloo
  <br>
  <small><sup>*</sup>Equal Contribution</small>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/YOUR_ARXIV_ID">
    <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat&labelColor=555555&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  &nbsp;
  <a href="YOUR_PROJECT_PAGE_URL">
    <img src="https://img.shields.io/badge/Project-Page-4F46E5?style=flat&labelColor=555555&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  &nbsp;
  <a href="#">
    <img src="https://img.shields.io/badge/Code-Coming%20Soon-FF5722?style=flat&labelColor=555555&logo=github&logoColor=white" alt="Code Coming Soon">
  </a>
</p>

</div>

---

## 📖 Abstract

Existing video generation models struggle to maintain long-term spatial and temporal consistency due to the dense, high-dimensional nature of video signals. To overcome this limitation, we propose **Spatia**, a spatial memory-aware video generation framework that explicitly preserves a 3D scene point cloud as persistent spatial memory. 

Spatia iteratively generates video clips conditioned on this spatial memory and continuously updates it through visual SLAM. This **dynamic-static disentanglement** design enhances spatial consistency throughout the generation process while preserving the model's ability to produce realistic dynamic entities. 

Furthermore, Spatia enables applications such as:
* **Explicit Camera Control**
* **3D-Aware Interactive Editing**
* **Long-horizon Scene Exploration**

<br>

<div align="center">
  <img src="./assets/teaser.png" width="100%" alt="Spatia Teaser"/>
</div>

---

<p align="center">
  <small>© 2024 Spatia Project. Licensed under <a href="http://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</small>
</p>