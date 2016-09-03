# -*- coding: utf-8 -*-
import wx
import wx.lib
import wx.lib.plot as plot

import math
import random as rnd

from chainer import cuda, optimizers, FunctionSet, Variable, Chain
import chainer.functions as F

import numpy as np
import copy

# import pickle

# Steps looking back
STATE_NUM = 2

# State
STATE_NUM = 2
NUM_EYES  = 9
STATE_DIM = NUM_EYES * 3 * 2 + 5 # 5 actions

class SState(object):
    def __init__(self):
        self.seq = np.ones((STATE_NUM, NUM_EYES*3), dtype=np.float32)
        
    def push_s(self, state):
        self.seq[1:STATE_NUM] = self.seq[0:STATE_NUM-1]
        self.seq[0] = state
        
    def fill_s(self, state):
        for i in range(0, STATE_NUM):
            self.seq[i] = state

class Q(Chain):
    def __init__(self, state_dim = STATE_DIM):
        super(Q, self).__init__(
            l1=F.Linear(state_dim, 50),
            l2=F.Linear(50, 50),
            q_value=F.Linear(50, 5)
        )
    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x, train=True), t)
        
    def predict(self, x, train = False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        y = self.q_value(h2)
        return y

class Walls(object):
    def __init__(self, x0, y0, x1, y1):
        self.xList = [x0, x1]
        self.yList = [y0, y1]
        self.P_color = wx.Colour(50,50,50)

    def addPoint(self, x, y):
        self.xList.append(x)
        self.yList.append(y)

    def Draw(self,dc):
        dc.SetPen(wx.Pen(self.P_color))
        for i in range(0, len(self.xList)-1):
            dc.DrawLine(self.xList[i], self.yList[i], self.xList[i+1],self.yList[i+1])

    def IntersectLine(self, p0, v0, i):
        dp = [p0[0] - self.xList[i], p0[1] - self.yList[i]]
        v1 = [self.xList[i+1] - self.xList[i], self.yList[i+1] - self.yList[i]]

        denom = float(v1[1]*v0[0] - v1[0]*v0[1])
        if denom == 0.0:
            return [False, 1.0]

        ua = (v1[0] * dp[1] - v1[1] * dp[0])/denom
        ub = (v0[0]*dp[1] - v0[1] * dp[0])/denom

        if 0 < ua and ua< 1.0 and 0 < ub and ub < 1.0:
            return [True, ua]

        return [False, 1.0]

    def IntersectLines(self, p0, v0):
        tmpt = 1.0
        tmpf = False
        for i in range(0, len(self.xList)-1):
            f,t = self.IntersectLine( p0, v0, i)
            if f:
                tmpt = min(tmpt, t)
                tmpf = True
 
        return [tmpf, tmpt]

class Ball(object):
    def __init__(self, x, y, color, property = 0):
        self.pos_x = x
        self.pos_y = y
        self.rad = 10 
        
        self.property = property

        self.B_color = color
        self.P_color = wx.Colour(50,50,50)

    def Draw(self, dc):
        dc.SetPen(wx.Pen(self.P_color))
        dc.SetBrush(wx.Brush(self.B_color))
        dc.DrawCircle(self.pos_x, self.pos_y, self.rad)
    
    def SetPos(self, x, y):
        self.pos_x = x
        self.pos_y = y
        
    def IntersectBall(self, p0, v0):
        # StackOverflow:Circle line-segment collision detection algorithm?
        # http://goo.gl/dk0yO1

        o = [-self.pos_x + p0[0], -self.pos_y + p0[1]]
                
        a = v0[0] ** 2 + v0[1] **2
        b = 2 * (o[0]*v0[0]+o[1]*v0[1])
        c = o[0] ** 2 + o[1] **2 - self.rad ** 2
        
        discriminant = float(b * b - 4 * a * c)
        
        if discriminant < 0:
            return [False, 1.0]
        
        discriminant = discriminant ** 0.5
        
        t1 = (- b - discriminant)/(2*a)
        t2 = (- b + discriminant)/(2*a)
        
        if t1 >= 0 and t1 <= 1.0:
            return [True, t1]

        if t2 >= 0 and t2 <= 1.0:
            return [True, t2]

        return [False, 1.0] 
        
class EYE(object):
    def __init__(self, i):
        self.OffSetAngle   = - math.pi/3 + i * math.pi*2/3/NUM_EYES
        self.SightDistance = 0
        self.obj           = -1
        self.FOV           = 130.0
        self.resetSightDistance()
        
    def resetSightDistance(self):
        self.SightDistance = self.FOV
        self.obj = -1

class Agent(Ball):
    def __init__(self, panel, x, y, epsilon = 0.99, model = None):
        super(Agent, self).__init__(
            x, y, wx.Colour(112,146,190)
        )
        self.dir_Angle = math.pi/4
        self.speed     = 5
        
        self.pos_x_max, self.pos_y_max = panel.GetSize()
        self.pos_y_max = 480
        
        self.eyes = [ EYE(i) for i in range(0, NUM_EYES)]
        
        self.actions = [[0.5, 0.0], [0.8, 1.0], [1.0, 1.0], [0.0, 0.5], [1.0, 0.8]]
        self.prevActions = np.zeros(len(self.actions))
        
        # DQN Model
        if model==None:
            self.model = Q()
            self.optimizer = optimizers.Adam()
            self.optimizer.setup(self.model)
            self.Primary = True
        else:
            self.model = model
            self.Primary = False
       
        self.epsilon = epsilon
        
        self.eMem = np.array([],dtype = np.float32)
        self.memPos = 0 
        self.memSize = 30000
        
        self.batch_num = 30
        self.gamma = 0.7
        self.loss = 0.0
        
        self.State = SState()
        self.prevState = np.ones((1,STATE_DIM))
    
    def UpdateState(self):
        s = np.ones((1, NUM_EYES * 3),dtype=np.float32)
        for i in range(0, NUM_EYES):
            if self.eyes[i].obj != -1:
                s[0, i * 3 + self.eyes[i].obj] = self.eyes[i].SightDistance / self.eyes[i].FOV
        self.State.push_s(s)

    def Draw(self, dc):
        dc.SetPen(wx.Pen(self.P_color))
        dc.SetBrush(wx.Brush(self.B_color))
        if self.Primary:
            for e in self.eyes:
                if e.obj == 1:
                    dc.SetPen(wx.Pen(wx.Colour(112,173,71)))
                elif e.obj == 2:
                    dc.SetPen(wx.Pen(wx.Colour(237,125,49)))
                else:
                    dc.SetPen(wx.Pen(self.P_color))
                dc.DrawLine(self.pos_x, self.pos_y, 
                        self.pos_x + e.SightDistance*math.sin(self.dir_Angle + e.OffSetAngle),
                        self.pos_y - e.SightDistance*math.cos(self.dir_Angle + e.OffSetAngle))
     
        super(Agent, self).Draw(dc)

    def get_action_value(self, state):
        x = Variable(state.reshape((1, -1)))
        return self.model.predict(x).data[0]
        
    def get_greedy_action(self, state):
        Q = self.get_action_value(state)
        action_index = np.argmax(Q)
        return action_index, Q
        
    def reduce_epsilon(self):
        self.epsilon -= 1.0/2000
        self.epsilon = max(0.02, self.epsilon) 
        
    def get_action(self,state,train):
        action = 0
        if train==True and np.random.random() < self.epsilon:
            action_index = np.random.randint(len(self.actions))
            Q = [0.0]
        else:
            action_index, Q = self.get_greedy_action(state)
        indices = np.zeros(len(self.actions))
        indices[action_index] = 1
        return self.actions[action_index], indices, Q

    def experience(self,x):
        if self.eMem.shape[0] > self.memSize:
            self.eMem[int(self.memPos%self.memSize)] = x
            self.memPos+=1
        elif self.eMem.shape[0] == 0:
            self.eMem = x
        else:       
            self.eMem = np.vstack( (self.eMem, x) )

    def update_model(self):
        if len(self.eMem)<self.batch_num:
            return

        memsize     = self.eMem.shape[0]
        batch_index = np.random.permutation(memsize)[:self.batch_num]
        batch       = np.array(self.eMem[batch_index], dtype=np.float32).reshape(self.batch_num, -1)

        x = Variable(batch[:,0:STATE_DIM])
        targets = self.model.predict(x).data.copy()

        for i in range(self.batch_num):
            #[ state..., action, reward, seq_new]
            a = int(batch[i,STATE_DIM])
            r = batch[i, STATE_DIM+1]

            new_seq= batch[i,(STATE_DIM+2):(STATE_DIM*2+2)]

            targets[i,a]=( r + self.gamma * np.max(self.get_action_value(new_seq)))

        t = Variable(np.array(targets, dtype=np.float32).reshape((self.batch_num,-1))) 

        # ネットの更新
        self.model.zerograds()
        loss=self.model(x ,t)
        self.loss = loss.data
        loss.backward()
        self.optimizer.update()
        
    def Move(self, WallsList):
        dp = [ self.speed * math.sin(self.dir_Angle),
              -self.speed * math.cos(self.dir_Angle)]
              
        for w in WallsList:
            if w.IntersectLines([self.pos_x, self.pos_y], dp)[0]:
                dp = [0.0, 0.0]

        self.pos_x += dp[0] 
        self.pos_y += dp[1]
   
        self.pos_x = max(0, min(self.pos_x, self.pos_x_max))
        self.pos_y = max(0, min(self.pos_y, self.pos_y_max))

    def HitBall(self, b):
        if ((b.pos_x - self.pos_x)**2+(b.pos_y - self.pos_y)**2)**0.5 < (self.rad + b.rad):
            return True
        return False

class World(wx.Frame):
    def __init__(self, parent=None, id=-1, title=None):
        wx.Frame.__init__(self, parent, id, title)
        self.MainPanel = wx.Panel(self, size=(640, 640))
        self.MainPanel.SetBackgroundColour('WHITE')
        
        self.panel = wx.Panel(self.MainPanel, size = (640,480))
        self.panel.SetBackgroundColour('WHITE')
        self.plotter = plot.PlotCanvas(self.MainPanel, size =(640, 640-480))
        self.plotter.SetEnableZoom(False)
        self.plotter.SetEnableLegend(True)
        self.plotter.SetFontSizeLegend(10.5)
        
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(self.panel)
        mainSizer.Add(self.plotter)

        self.SetSizer(mainSizer)
        self.Fit()
 
        self.A = Agent(self.panel, 150, 100 )
        self.B = Agent(self.panel, 400, 300, model = self.A.model)
        self.C = Agent(self.panel, 400, 150, model = self.A.model)
        self.D = Agent(self.panel, 320, 240, model = self.A.model)
        
        self.greenB = [Ball(rnd.randint(40, 600),rnd.randint(40, 440), 
                        wx.Colour(112,173,71), property =  1) for i in range(0, 15)]
        self.redB  = [Ball(rnd.randint(40, 600),rnd.randint(40, 440), 
                        wx.Colour(237,125,49), property = 2) for i in range(0, 10)]
         
        # OutrBox
        self.Box = Walls(640, 480, 0, 480)
        self.Box.addPoint(0,0)
        self.Box.addPoint(640,0)
        self.Box.addPoint(640,480)
        
        # Wall in the world
        self.WallA = Walls(96, 90, 256, 90)
        self.WallA.addPoint(256, 390)
        self.WallA.addPoint(96,390)
        
        self.Bind(wx.EVT_CLOSE, self.CloseWindow)
 
        self.cdc = wx.ClientDC(self.panel)
        w, h = self.panel.GetSize()
        self.bmp = wx.EmptyBitmap(w,h)
 
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer)
        self.timer.Start(20)
        
        self.i = 0
        self.tmp_sum = 0.0
        self.data = []

    def CloseWindow(self, event):
        # self.timer.Stop()
        wx.Exit()
 
    def OnTimer(self, event):
        for ag in [self.A, self.B, self.C, self.D]:
            # Update States
            for e in ag.eyes:
                e.resetSightDistance()
                p = [ag.pos_x, ag.pos_y]
                s = math.sin(ag.dir_Angle + e.OffSetAngle)
                c = math.cos(ag.dir_Angle + e.OffSetAngle)
                
                for g in self.greenB:
                    f, t = g.IntersectBall(p, [e.SightDistance * s, - e.SightDistance * c])
                    if f:
                        e.SightDistance *= t
                        e.obj = g.property
                        
                for r in self.redB:
                    f, t = r.IntersectBall(p, [e.SightDistance * s, - e.SightDistance * c])
                    if f:
                        e.SightDistance *= t
                        e.obj = r.property

                for w in [self.Box, self.WallA]:
                    f, t = w.IntersectLines(p, [e.SightDistance * s, - e.SightDistance * c])
                    if f:
                        e.SightDistance *= t
                        e.obj = 0
                        
                
            # Update States
            ag.UpdateState()
            state = np.hstack((ag.State.seq.reshape(1,-1), 
                        ag.prevActions.reshape(1,-1))).astype(np.float32)
            action, ag.prevActions, Q = ag.get_action(state, True)
            
            # Action Step
            ag.speed = (action[0]+action[1])/2.0 * 10.0
            ag.dir_Angle += math.atan((action[0] - action[1]) * 10.0 / 2.0 / 5.0)
            ag.dir_Angle = ( ag.dir_Angle + np.pi) % (2 * np.pi ) - np.pi
            ag.Move([self.Box, self.WallA])
            digestion_reward = 0.0        
            for g in self.greenB:
                if ag.HitBall(g):
                    g.SetPos(rnd.randint(40, 600),rnd.randint(40, 440))
                    digestion_reward -= 6.0
            for r in self.redB:
                if ag.HitBall(r):
                    r.SetPos(rnd.randint(40, 600),rnd.randint(40, 440))
                    digestion_reward += 5.0

            # Reward
            proximity_reward = 0.0
            for e in ag.eyes:
                proximity_reward += float(e.SightDistance)/e.FOV if e.obj == 0 else 1.0
            proximity_reward /= NUM_EYES
            proximity_reward = min(1.0, proximity_reward*2)
            
            forward_reward   = 0.0
            if(action[0] == action[1] and proximity_reward > 0.75):  
                forward_reward = 0.1 * proximity_reward
                
            reward = proximity_reward + forward_reward + digestion_reward
        
            # Learning Step [Shared Ex Memory]
            self.A.experience(np.hstack([
                        self.A.prevState,
                        np.array([np.argmax(ag.prevActions)]).reshape(1,-1),
                        np.array([reward]).reshape(1,-1),
                        state
                    ]))
            ag.prevState = state.copy()
        # Do Reinforcement Learning
        self.A.update_model()
        self.A.reduce_epsilon()

        # Graphics Update
        for e in self.A.eyes:
            e.resetSightDistance()
            p = [self.A.pos_x, self.A.pos_y]
            s = math.sin(self.A.dir_Angle + e.OffSetAngle)
            c = math.cos(self.A.dir_Angle + e.OffSetAngle)
            
            for g in self.greenB:
                f, t = g.IntersectBall(p, [e.SightDistance * s, - e.SightDistance * c])
                if f:
                    e.SightDistance *= t
                    e.obj = g.property

            for r in self.redB:
                f, t = r.IntersectBall(p, [e.SightDistance * s, - e.SightDistance * c])
                if f:
                    e.SightDistance *= t
                    e.obj = r.property

            for w in [self.Box, self.WallA]:
                f, t = w.IntersectLines(p, [e.SightDistance * s, - e.SightDistance * c])
                if f:
                    e.SightDistance *= t
                    e.obj = 0

        self.bdc = wx.BufferedDC(self.cdc, self.bmp)
        self.gcdc = wx.GCDC(self.bdc)
        self.gcdc.Clear()
        
        self.gcdc.SetPen(wx.Pen('white'))
        self.gcdc.SetBrush(wx.Brush('white'))
        self.gcdc.DrawRectangle(0,0,640,640)

        for ag in [self.A, self.B, self.C, self.D]:
            ag.Draw(self.gcdc)
        for g in self.greenB:
            g.Draw(self.gcdc)
        for r in self.redB:
            r.Draw(self.gcdc)
            
        self.Box.Draw(self.gcdc)
        self.WallA.Draw(self.gcdc)
        # 学習記録
        self.tmp_sum += reward

        if self.i % 50 == 1:
            self.data.append((self.i, self.tmp_sum/50.0))
            self.tmp_sum = 0.0

            if len(self.data) >= 3000:
                self.data = self.data[-3001:]

            line = plot.PolyLine(self.data, legend='Reward', colour='red', width = 1)
            self.gc = plot.PlotGraphics([line], 'Reward', 'Steps', 'Reward per Sec')
            self.plotter.Draw(self.gc,
                xAxis=(max(0,self.i-3000), self.i), 
                yAxis=(0, 1.5))
                
            print 'Epsilon:', self.A.epsilon
        self.i += 1

 
if __name__ == '__main__':
    app = wx.PySimpleApp()
    w = World(title='RL Test')
    w.Center()
    w.Show()
    app.MainLoop()
