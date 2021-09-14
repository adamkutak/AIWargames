import numpy as np
import math
import nnMatrix as nn
import turtle
import time
import copy

# ----------------------------------------------------------------
# game rules section
# ----------------------------------------------------------------
#20 by 20 square grid
#4 teams: red, blue, green, yellow
#each team starts with a 4 by 4 area
#all territory owned by a player must be connected

#turns work as follows:
# 1. each player simultaneously decides where to put their square
# 2. squares are placed down
# 3. if a player has more than 1 connected area, the largest
#    one persists, and the other ones dissapear/absorb based
#    on the following rules:
#     1. if no other players have adjacent territory, that territory
#        is turned blank (becomes unoccupied)
#     2. otherwise, the player that covers the most perimeter of the rogue area
#        absorbs that area as part of its territory

#we play 1,000 turns.
# ----------------------------------------------------------------
#issues:
# 1. AI has trouble learning from scratch when initially placing squares
# solution: create a preset training game that will guide them into placing
#           blocks in the correct places, then once they know how to expand
#           their territory, they can begin to "fight" eachother proper!
# ----------------------------------------------------------------

#set constants
# ----------------------------------------------------------------
teams = 4
grid_height = 20
grid_length = 20
start_height = 4
start_length = 4
turns = 100
# ----------------------------------------------------------------

#team placement
# ----------------------------------------------------------------
def place_teams(mapp):

    #team 1
    for x in range(start_length):
        for y in range(start_height):
            mapp[x][y] = 1
    #team 2
    for x in range(start_length):
        for y in range(start_height):
            mapp[grid_length - 1 - x][y] = 2
    #team 3
    for x in range(start_length):
        for y in range(start_height):
            mapp[x][grid_height - 1 - y] = 3
    #team 4
    for x in range(start_length):
        for y in range(start_height):
            mapp[grid_length - 1 - x][grid_height - 1 - y] = 4

    return
# ----------------------------------------------------------------
#player functions
# ----------------------------------------------------------------
def place_block(p,curr_mapp,player,input_AI,mapp):
    #make decision HERE
    n = input_AI[player-1].calculate(curr_mapp.flatten())
    p = get_perim(get_all_terr(player,mapp),player,mapp)
    x,y = grid(select_square(p,n))
    mapp[x][y] = player
    return

# ----------------------------------------------------------------
#screen draw functions
# ----------------------------------------------------------------
def update_draw(mapp,tur):
    tur.clear()
    col = ["white","red","green","blue","yellow"]
    tur.setpos(0,0)
    xsize = 20
    ysize = 20
    for x in range(grid_length):
        for y in range(grid_height):
            tur.fillcolor(col[int(mapp[x][y])])
            tur.setpos(x*xsize,y*ysize)
            tur.begin_fill()
            for _ in range(4):
                tur.forward(xsize)
                tur.right(90)
            tur.end_fill()
    turtle.update()
    return
# ----------------------------------------------------------------
#misc functions
# ----------------------------------------------------------------
#conversion functions
def numd(x,y):
    return (grid_length*y + x)

def grid(num):
    x = num%grid_length
    y = int((num-x)/grid_length)
    return x,y

#test for 2 blocks are adjacent
def adjt(a,b):
    if(type(a)!=tuple):
        x,y = grid(a)
        a = (x,y)
    if(type(b)!=tuple):
        x2,y2 = grid(b)
        b = (x2,y2)
    if(abs(a[0]-b[0])==1 and a[1]==b[1]):
        return True
    elif(abs(a[1]-b[1])==1 and a[0]==b[0]):
        return True
    return False

#get all territory of player
def get_all_terr(player,mapp):
    terr = []
    for q in range(grid_height*grid_length):
        x,y = grid(q)
        if(mapp[x][y]==player):
            terr.append(q)
    return terr

#gets the different chunks of player territory
def get_terr_chunks(player,mapp):
    terr = get_all_terr(player,mapp)
    ct = []
    cc = 0
    while(len(terr) > 0):
        hold = []
        hold.append(terr.pop(0))
        ct.append([])
        while(len(hold) > 0):
            hold = get_surround(hold,ct,cc,player,terr,mapp)
            ct[cc].append(hold.pop(0))
        cc += 1
    return ct

#helper function:
#gets the surrounding blocks of the first element of hold, adds them to hold (doesn't add duplicates)
def get_surround(hold,ct,cc,player,terr,mapp):
    schema = [(1,0),(-1,0),(0,1),(0,-1)] #these are the 4 possible adjacency schema: up, down, left, right
    x,y = grid(hold[0])
    for s in schema:
        xt,yt = x + s[0], y + s[1]
        #check for in-bound of mapp
        if(xt < grid_length and xt > -1 and yt < grid_height and yt > -1):
            if(mapp[xt][yt]==player):
                #check for not duplicate
                n = numd(xt,yt)
                if(n not in hold and n not in ct[cc]):
                    #finally, we know it is valid so add to hold, remove from terr
                    hold.append(n)
                    terr.pop(terr.index(n))
    return hold

def biggest_terr(ct):
    len_ind = []
    for w in ct:
        len_ind.append(len(w))
    return len_ind.index(max(len_ind))

def get_perim(terr,player,mapp):
    perim = []
    schema = [(1,0),(-1,0),(0,1),(0,-1)]
    for t in terr:
        x,y = grid(t)
        for s in schema:
            xt,yt = x + s[0], y + s[1]
            if(xt < grid_length and xt > -1 and yt < grid_height and yt > -1):
                if(mapp[xt][yt]!=player):
                    n = numd(xt,yt)
                    if(n not in perim):
                        perim.append(n)
    return perim

def perim_comp(perim,mapp):
    comp = [0,0,0,0,0]
    for p in perim:
        x,y = grid(p)
        comp[int(mapp[x][y])] += 1
    if(sum(comp)-comp[0]==0):
        return 0
    else:
        return comp.index(max(comp[1:]))

def select_square(perim,temp):
    n = np.argmax(temp)
    while(n not in perim):
        temp[n] = 0
        n = np.argmax(temp)
    return n
# ----------------------------------------------------------------
#game loop
# ----------------------------------------------------------------
def play(input_AI,turns):
    # turtle.tracer(0,0)
    # s = turtle.getscreen()
    # s.setup(800,800)
    # tur = turtle.Turtle()
    # tur.speed(0)
    mapp = np.zeros((grid_height,grid_length))
    place_teams(mapp)
    #update_draw(mapp,tur)
    #per turn game loop
    for l in range(turns):

        #each team places a block, with knowledge of the current mapp
        curr_mapp = mapp
        for t in range(teams):
            player = t+1
            p = get_perim(get_all_terr(player,mapp),player,mapp)
            place_block(p,curr_mapp,player,input_AI,mapp)

        #resolve multiple territory issues
        for t in range(teams):
            player = t+1
            ct = get_terr_chunks(player,mapp)
            i = biggest_terr(ct)

            #decide what to do with the smaller territories
            for c in range(len(ct)):
                if(c!=i):
                    #calculate the perimeter composition and assign territory if there is a split
                    perim = get_perim(ct[c],player,mapp)
                    winner = perim_comp(perim,mapp)
                    for b in ct[c]:
                        x,y = grid(b)
                        mapp[x][y] = winner
        #update_draw(mapp,tur)

    #game is over, print the mapp
    #print(mapp)
    return [len(get_all_terr(1,mapp)),len(get_all_terr(2,mapp)),len(get_all_terr(3,mapp)),len(get_all_terr(4,mapp))]
# ----------------------------------------------------------------

#run the game normally
# ----------------------------------------------------------------
# #make blank AI
# input_AI = []
# for t in range(teams):
#     input_AI.append(nn.Network(str(t),grid_height*grid_length,grid_height*grid_length,[200]))
#
# #play game
# play(input_AI)
# ----------------------------------------------------------------
#new addition: pregenetic training
#we will calculate correct output training data on the fly as we feed each AI team the input
#the correct output is a 1 for every adjacent square to the AI's territory, i.e the
#AI should learn to place a square where it will expand its territory, therefore acting efficiently.
#this will be a modified "play" function, which will then save the AI to a file!

def play_mod(input_AI,turns):
    #setup for drawing
    turtle.tracer(0,0)
    s = turtle.getscreen()
    s.setup(800,800)
    tur = turtle.Turtle()
    tur.speed(0)

    mapp = np.zeros((grid_height,grid_length))
    place_teams(mapp)

    #update grid
    update_draw(mapp,tur)

    #per turn game loop
    for l in range(turns):

        #each team places a block, with knowledge of the current mapp
        curr_mapp = mapp
        for t in range(teams):
            player = t+1
            p = get_perim(get_all_terr(player,mapp),player,mapp)
            perimmap = np.zeros((grid_height,grid_length))
            #making correct answers all the perim:
            for sq in p:
                x,y = grid(sq)
                perimmap[x][y] = 1

            #OR: just 1 square of perim is correct:
            #x,y = grid(p[0])
            #perimmap[x][y] = 1

            #print(perimmap)
            temp = input_AI[player-1].onTheFlyTeach(curr_mapp.flatten(),perimmap.flatten())
            #if(t==2):
                #print(np.round(temp,1))
                #print(perimmap.flatten())
            n = select_square(p,temp)
            #print(t)
            #print(curr_mapp.flatten())
            #print(perimmap.flatten())
            x,y = grid(n)
            mapp[x][y] = player

        #resolve multiple territory issues
        for t in range(teams):
            player = t+1
            ct = get_terr_chunks(player,mapp)
            i = biggest_terr(ct)

            #decide what to do with the smaller territories
            for c in range(len(ct)):
                if(c!=i):
                    #calculate the perimeter composition and assign territory if there is a split
                    perim = get_perim(ct[c],player,mapp)
                    winner = perim_comp(perim,mapp)
                    for b in ct[c]:
                        x,y = grid(b)
                        mapp[x][y] = winner
        #print(mapp)
        update_draw(mapp,tur)
        print(l)
        #print(len(get_all_terr(1,mapp)))
    #game is over, print the mapp and save the 4 AI to seperate files!

    for t in range(teams):
        input_AI[t].save("war-"+str(t))
    return [len(get_all_terr(1,mapp)),len(get_all_terr(2,mapp)),len(get_all_terr(3,mapp)),len(get_all_terr(4,mapp))]

# ----------------------------------------------------------------
#run the pregenetic training
# ----------------------------------------------------------------
def pregenTrain(turns,loader):
    trainAI = []
    for t in range(2):
        trainAI.append(nn.Network(str(t),grid_height*grid_length,grid_height*grid_length,[(grid_height*grid_length)//2,200]))
        trainAI[-1].load(loader+str(t))
    for t in range(2,4):
        trainAI.append(nn.Network(str(t),grid_height*grid_length,grid_height*grid_length,[(grid_height*grid_length)//2,200]))
        trainAI[-1].load("war-"+str(t))
    play_mod(trainAI,turns)
    return
# ----------------------------------------------------------------
#run the game with the AI genetic algorithm
# ----------------------------------------------------------------
def runGenetic(turns,gen_cap,generations,loader,saver):
    #take the 4 pretrained AI, make 49 children of each.
    AI = []
    for t in range(teams):
        temp_AI = []
        ta = nn.Network(str(t),grid_height*grid_length,grid_height*grid_length,[(grid_height*grid_length)//2,200])
        ta.load(loader+str(t))
        temp_AI.append(ta)
        for g in range(gen_cap-1):
            temp_AI.append(nn.Network(str(t),grid_height*grid_length,grid_height*grid_length,[(grid_height*grid_length)//2,200],True,temp_AI[0].w,temp_AI[0].b))
        AI.append(temp_AI)

    #run the game, selecting the best AI each time to replicate
    pab = [0]*4
    for gen in range(generations):
        area = [0]*4
        ind = [0]*4
        for a in range(gen_cap):
            input_AI = [AI[0][a],AI[1][a],AI[2][a],AI[3][a]]
            rt = play(input_AI,turns)
            for t in range(teams):
                if(rt[t] > area[t]):
                    ind[t] = a
                    area[t] = rt[t]

        print("gen: ",gen,", best = ",area)
        #reproduce
        best = []
        for x in range(teams):
            if(area[x] >= pab[x]):
                best.append(AI[x][ind[x]])
            else:
                best.append(pb[x])
        pb = copy.deepcopy(best)
        pab = copy.deepcopy(area)
        AI = []
        for t in range(teams):
            temp_AI = []
            temp_AI.append(best[t])
            for g in range(gen_cap-1):
                temp_AI.append(nn.Network(str(t),grid_height*grid_length,grid_height*grid_length,[(grid_height*grid_length)//2,200],True,best[t].w,best[t].b))
            AI.append(temp_AI)

    #save the final AI
    for t in range(teams):
        best[t].save(saver+str(t))
#actual run section:
pregenTrain(10000,"warG-")
#runGenetic(10000,50,50,"war-","warG-")
