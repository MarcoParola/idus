# idus

[![license](https://img.shields.io/static/v1?label=OS&message=Linux&color=green&style=plastic)]()
[![Python](https://img.shields.io/static/v1?label=Python&message=3.10&color=blue&style=plastic)]()


Object detection design framework supporting:
- Yuo Only Look Once (YOLO)
- DEtection TRanformer (DETR)
- You Only Look at One Sequence (YOLOS)

# TODO 
- integrare YOLO
- può aver senso modificare l'attuale computazione della loss che viene fatta in [criterion.py](./src/models/criterion.py) 
Per due motivi: (i) dovremo definire alcune varianti di loss per alcuni metodi di unlearning. (ii) vogliamo disaccoppiare quello che è il calcolo delle loss dal calcolo delle metriche, perchè la loss è specifica per i transformer, ma le metriche idealmente le vogliamo riutilizzare pari pari per yolo. 
Forse lo farei così: 1. fare una copia-incollare SetCriterion nello stesso file e uno dei due rinominarlo "BaseCriterion" mentre l'altro "ObjDetectionMetrics". Dal primo levi tutta la parte delle metriche, dal secondo delle loss. Entrambi continuano a restituire un dizionario. In `train.py` instanzi due differenti variabili `criterion=BaseCriterion(..)` e `metrics=ObjDetectionMetrics(..)` .

## Acknowledgement
Special thanks to [@clive819](https://github.com/clive819) for making an implementation of DETR public [here](https://github.com/clive819/Modified-DETR). Special thanks to [@hustvl](https://github.com/hustvl) for YOLOS [original implementation](https://github.com/hustvl/YOLOS)
