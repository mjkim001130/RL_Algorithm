# RL_code

pytorch 튜토리얼의 코드를 참고함

[Pytorch Tutorial](https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html)

---
# In Cartpole

>### DQN

<img width="443" alt="image" src="https://github.com/mjkim001130/RL_code/assets/79756320/fe0fdac8-cb42-4465-a4a1-4159de13cb33">

튜토리얼 가이드에 따르면 600회까지 학습을 했을때 500에 수렴하는 결과를 보여야 한다고 한다. 그게 아닐시에 재학습 추천

>###DQN_POMDP

<img width="427" alt="image" src="https://github.com/mjkim001130/RL_code/assets/79756320/29f1d148-121a-4d60-83ee-beba4fc9cea8">

카트의 위치와 각도만을 부분적으로 관측(POMDP)한 상태를 만들어 학습함
DQN보다 진행속도 빠름, 결과는 보이는것처럼 아예 학습을 못함
