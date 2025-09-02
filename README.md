# air_quality_index_project

Run the below command to setup environment on Google Colab

```
WORKSPACE="/le_thanh_van_118/workspace/hiep_workspace" REPO="air_quality_index_project" && \
mkdir -p "$WORKSPACE" && \
cd "$WORKSPACE" && \
git clone "https://github.com/tphiepbk/$REPO.git" && \
cd "$REPO" && \
pushd "dataset" && \
bash downloader.sh && \
popd
```
