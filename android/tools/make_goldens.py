"""Regenerates app/src/{test/resources,androidTest/assets}/golden_pipeline.json from the Python PoC.

Mirrors the app default pipeline (detector 320 -> rec_71 corners -> quad warp -> rec_57, plus region
classifiers) with Pillow, so the Kotlin port can be checked against it.
Usage: python android/tools/make_goldens.py > golden_pipeline.json   (needs MNN, onnxruntime, numpy, pillow)
"""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import read_plate as rp, decode_samples as ds
from PIL import Image
det=rp.Detector(); corners=ds.Net("71"); c57=rp.Crnn("57")
latin={n:rp.Crnn(str(n)) for n in (50,53,57,60,63,65,66,68,69)}
reg={m:ds.Net(m) for m in ("23","26","29","32","35","38","41","44","47","82")}
ids={"23":"24","26":"27","29":"30","32":"33","35":"36","38":"39","41":"42","44":"45","47":"48","82":"83"}
def lines(n): return [l for l in open(rp.path(n),encoding="utf-8").read().splitlines() if l.strip()]
out={}
for f in ("BRL4104","EV232","PUTRAJAYA541"):
  img=Image.open(os.path.join(ROOT, "sample_data", f + ".jpeg")).convert("RGB")
  w,h=img.size; r=320/max(w,h); nw,nh=round(w*r),round(h*r)
  small=np.asarray(img.resize((nw,nh),Image.BILINEAR,reducing_gap=2.0))
  canvas=np.full((320,320,3),114,np.uint8); canvas[:nh,:nw]=small
  lb_sum=float(canvas.astype(np.float64).sum()/255)
  x1,y1,x2,y2,s=det(img)
  t_plain,_=rp.crop_plate(img,(x1,y1,x2,y2),0.0) # pad .02 module const
  rp.CROP_PAD=0.03; t_p03,_=rp.crop_plate(img,(x1,y1,x2,y2),0.0); rp.CROP_PAD=0.02
  t_shear,_=rp.crop_plate(img,(x1,y1,x2,y2),0.3)
  bw,bh=x2-x1,y2-y1; cx1,cy1,cx2,cy2=x1-.2*bw,y1-.5*bh,x2+.2*bw,y2+.5*bh
  c=img.crop((cx1,cy1,cx2,cy2)).resize((96,48),Image.BILINEAR)
  a=np.asarray(c,np.float32)[...,::-1]/127.5-1
  p=corners(np.ascontiguousarray(a.transpose(2,0,1)[None]))
  rx1,ry1=round(cx1),round(cy1); rw,rh=round(cx2)-rx1,round(cy2)-ry1
  q=[rx1+p[i]*rw if i%2==0 else ry1+p[i]*rh for i in range(8)]
  ctr=np.array([np.mean(q[0::2]),np.mean(q[1::2])])
  qq=[]
  for i in range(4): qq+=list(ctr+(np.array(q[2*i:2*i+2])-ctr)*1.03)
  warped=img.transform((192,96),Image.QUAD,tuple(float(v) for v in qq),Image.BILINEAR)
  in96=np.asarray(warped.resize((96,48),Image.BILINEAR),np.float32)
  t_rect=np.ascontiguousarray(in96[...,::-1].transpose(2,0,1)[None]/255)
  raw=np.ascontiguousarray(in96[...,::-1].transpose(2,0,1)[None])
  regions={}
  for m,n in ids.items():
    pr=ds.softmax_if_logits(reg[m](raw)); k=int(pr.argmax()); regions[m]=[lines(n)[k],round(float(pr[k]),3)]
  out[f]={"letterbox_sum":lb_sum,"box":[float(x1),float(y1),float(x2),float(y2)],"det_score":float(s),
    "plain_pad02_sum":float(t_plain.sum()),"plain_pad03_sum":float(t_p03.sum()),"shear03_pad02_sum":float(t_shear.sum()),
    "corners":[float(v) for v in q],"rect_in96_sum":float(t_rect.sum()),
    "rect_reads":{n:list(m(t_rect)) for n,m in latin.items()},"poc_read":list(c57(t_shear)),"regions":regions}
print(json.dumps(out,indent=1))
