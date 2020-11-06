## CharBERT: Character-aware Pre-trained Language Model

Wentao Ma et al. (iFLYTEK Research)



### References

- CharBERT: [COLING 2020](https://arxiv.org/abs/2011.01513)
- [repository](https://github.com/wtma/CharBERT)



### Summary

- 서브워드는 불완전하고 깨지기 쉬운 Representation을 학습
- 캐릭터와 서브워드를 함께 활용하는 방안을 제시해 볼 것
- 캐릭터를 활용하는 훈련 기법이 철자 오류에 강건한 PLM을 학습할 수 있음을 결과로 증명


### Introduction

- 서브워드 기반의 토크나이저는 거의 모든 단어를 인코딩하여, OOV 문제에서 비교적 자유롭다는 장점
- 그러나 서브워드 기반 언어 모델은 아래와 같은 단점을 지니고 있음
	- **Incomplete Modeling** : 캐릭터와 같은 _fine-grained_ 자질의 정보가 부족
	- **Fragile Representation** : 작은 오탈자 하나가 서브워드 분절을 망칠 수 있음
- 

