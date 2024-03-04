# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.


import numpy as np
import cv2 as cv

class PPResNet:
    def __init__(self, modelPath, topK=1, loadLabel=True, backendId=0, targetId=0):
        self._modelPath = modelPath
        assert topK >= 1
        self._topK = topK
        self._load_label = loadLabel
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.dnn.readNet(self._modelPath)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        self._inputNames = ''
        self._outputNames = ['save_infer_model/scale_0.tmp_0']
        self._inputSize = [224, 224]
        self._mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
        self._std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]

        # load labels
        self._labels = self._load_labels()

    def _load_labels(self):
        return self.LABELS_IMAGENET_1K.splitlines()

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image):
        image = image.astype(np.float32, copy=False) / 255.0
        image -= self._mean
        image /= self._std
        return cv.dnn.blobFromImage(image)

    def infer(self, image):
        assert image.shape[0] == self._inputSize[1], '{} (height of input image) != {} (preset height)'.format(image.shape[0], self._inputSize[1])
        assert image.shape[1] == self._inputSize[0], '{} (width of input image) != {} (preset width)'.format(image.shape[1], self._inputSize[0])

        # Preprocess
        inputBlob = self._preprocess(image)

        # Forward
        self._model.setInput(inputBlob, self._inputNames)
        outputBlob = self._model.forward(self._outputNames)

        # Postprocess
        results = self._postprocess(outputBlob[0])

        return results

    def _postprocess(self, outputBlob):
        batched_class_id_list = []
        for ob in outputBlob:
            class_id_list = ob.argsort()[::-1][:self._topK]
            batched_class_id_list.append(class_id_list)
        if len(self._labels) > 0 and self._load_label:
            batched_predicted_labels = []
            for class_id_list in batched_class_id_list:
                predicted_labels = []
                for class_id in class_id_list:
                    predicted_labels.append(self._labels[class_id])
                batched_predicted_labels.append(predicted_labels)
            return batched_predicted_labels
        else:
            return batched_class_id_list

    LABELS_IMAGENET_1K = '''tench
goldfish
great white shark
tiger shark
hammerhead
electric ray
stingray
cock
hen
ostrich
brambling
goldfinch
house finch
junco
indigo bunting
robin
bulbul
jay
magpie
chickadee
water ouzel
kite
bald eagle
vulture
great grey owl
European fire salamander
common newt
eft
spotted salamander
axolotl
bullfrog
tree frog
tailed frog
loggerhead
leatherback turtle
mud turtle
terrapin
box turtle
banded gecko
common iguana
American chameleon
whiptail
agama
frilled lizard
alligator lizard
Gila monster
green lizard
African chameleon
Komodo dragon
African crocodile
American alligator
triceratops
thunder snake
ringneck snake
hognose snake
green snake
king snake
garter snake
water snake
vine snake
night snake
boa constrictor
rock python
Indian cobra
green mamba
sea snake
horned viper
diamondback
sidewinder
trilobite
harvestman
scorpion
black and gold garden spider
barn spider
garden spider
black widow
tarantula
wolf spider
tick
centipede
black grouse
ptarmigan
ruffed grouse
prairie chicken
peacock
quail
partridge
African grey
macaw
sulphur-crested cockatoo
lorikeet
coucal
bee eater
hornbill
hummingbird
jacamar
toucan
drake
red-breasted merganser
goose
black swan
tusker
echidna
platypus
wallaby
koala
wombat
jellyfish
sea anemone
brain coral
flatworm
nematode
conch
snail
slug
sea slug
chiton
chambered nautilus
Dungeness crab
rock crab
fiddler crab
king crab
American lobster
spiny lobster
crayfish
hermit crab
isopod
white stork
black stork
spoonbill
flamingo
little blue heron
American egret
bittern
crane
limpkin
European gallinule
American coot
bustard
ruddy turnstone
red-backed sandpiper
redshank
dowitcher
oystercatcher
pelican
king penguin
albatross
grey whale
killer whale
dugong
sea lion
Chihuahua
Japanese spaniel
Maltese dog
Pekinese
Shih-Tzu
Blenheim spaniel
papillon
toy terrier
Rhodesian ridgeback
Afghan hound
basset
beagle
bloodhound
bluetick
black-and-tan coonhound
Walker hound
English foxhound
redbone
borzoi
Irish wolfhound
Italian greyhound
whippet
Ibizan hound
Norwegian elkhound
otterhound
Saluki
Scottish deerhound
Weimaraner
Staffordshire bullterrier
American Staffordshire terrier
Bedlington terrier
Border terrier
Kerry blue terrier
Irish terrier
Norfolk terrier
Norwich terrier
Yorkshire terrier
wire-haired fox terrier
Lakeland terrier
Sealyham terrier
Airedale
cairn
Australian terrier
Dandie Dinmont
Boston bull
miniature schnauzer
giant schnauzer
standard schnauzer
Scotch terrier
Tibetan terrier
silky terrier
soft-coated wheaten terrier
West Highland white terrier
Lhasa
flat-coated retriever
curly-coated retriever
golden retriever
Labrador retriever
Chesapeake Bay retriever
German short-haired pointer
vizsla
English setter
Irish setter
Gordon setter
Brittany spaniel
clumber
English springer
Welsh springer spaniel
cocker spaniel
Sussex spaniel
Irish water spaniel
kuvasz
schipperke
groenendael
malinois
briard
kelpie
komondor
Old English sheepdog
Shetland sheepdog
collie
Border collie
Bouvier des Flandres
Rottweiler
German shepherd
Doberman
miniature pinscher
Greater Swiss Mountain dog
Bernese mountain dog
Appenzeller
EntleBucher
boxer
bull mastiff
Tibetan mastiff
French bulldog
Great Dane
Saint Bernard
Eskimo dog
malamute
Siberian husky
dalmatian
affenpinscher
basenji
pug
Leonberg
Newfoundland
Great Pyrenees
Samoyed
Pomeranian
chow
keeshond
Brabancon griffon
Pembroke
Cardigan
toy poodle
miniature poodle
standard poodle
Mexican hairless
timber wolf
white wolf
red wolf
coyote
dingo
dhole
African hunting dog
hyena
red fox
kit fox
Arctic fox
grey fox
tabby
tiger cat
Persian cat
Siamese cat
Egyptian cat
cougar
lynx
leopard
snow leopard
jaguar
lion
tiger
cheetah
brown bear
American black bear
ice bear
sloth bear
mongoose
meerkat
tiger beetle
ladybug
ground beetle
long-horned beetle
leaf beetle
dung beetle
rhinoceros beetle
weevil
fly
bee
ant
grasshopper
cricket
walking stick
cockroach
mantis
cicada
leafhopper
lacewing
dragonfly
damselfly
admiral
ringlet
monarch
cabbage butterfly
sulphur butterfly
lycaenid
starfish
sea urchin
sea cucumber
wood rabbit
hare
Angora
hamster
porcupine
fox squirrel
marmot
beaver
guinea pig
sorrel
zebra
hog
wild boar
warthog
hippopotamus
ox
water buffalo
bison
ram
bighorn
ibex
hartebeest
impala
gazelle
Arabian camel
llama
weasel
mink
polecat
black-footed ferret
otter
skunk
badger
armadillo
three-toed sloth
orangutan
gorilla
chimpanzee
gibbon
siamang
guenon
patas
baboon
macaque
langur
colobus
proboscis monkey
marmoset
capuchin
howler monkey
titi
spider monkey
squirrel monkey
Madagascar cat
indri
Indian elephant
African elephant
lesser panda
giant panda
barracouta
eel
coho
rock beauty
anemone fish
sturgeon
gar
lionfish
puffer
abacus
abaya
academic gown
accordion
acoustic guitar
aircraft carrier
airliner
airship
altar
ambulance
amphibian
analog clock
apiary
apron
ashcan
assault rifle
backpack
bakery
balance beam
balloon
ballpoint
Band Aid
banjo
bannister
barbell
barber chair
barbershop
barn
barometer
barrel
barrow
baseball
basketball
bassinet
bassoon
bathing cap
bath towel
bathtub
beach wagon
beacon
beaker
bearskin
beer bottle
beer glass
bell cote
bib
bicycle-built-for-two
bikini
binder
binoculars
birdhouse
boathouse
bobsled
bolo tie
bonnet
bookcase
bookshop
bottlecap
bow
bow tie
brass
brassiere
breakwater
breastplate
broom
bucket
buckle
bulletproof vest
bullet train
butcher shop
cab
caldron
candle
cannon
canoe
can opener
cardigan
car mirror
carousel
carpenters kit
carton
car wheel
cash machine
cassette
cassette player
castle
catamaran
CD player
cello
cellular telephone
chain
chainlink fence
chain mail
chain saw
chest
chiffonier
chime
china cabinet
Christmas stocking
church
cinema
cleaver
cliff dwelling
cloak
clog
cocktail shaker
coffee mug
coffeepot
coil
combination lock
computer keyboard
confectionery
container ship
convertible
corkscrew
cornet
cowboy boot
cowboy hat
cradle
crane
crash helmet
crate
crib
Crock Pot
croquet ball
crutch
cuirass
dam
desk
desktop computer
dial telephone
diaper
digital clock
digital watch
dining table
dishrag
dishwasher
disk brake
dock
dogsled
dome
doormat
drilling platform
drum
drumstick
dumbbell
Dutch oven
electric fan
electric guitar
electric locomotive
entertainment center
envelope
espresso maker
face powder
feather boa
file
fireboat
fire engine
fire screen
flagpole
flute
folding chair
football helmet
forklift
fountain
fountain pen
four-poster
freight car
French horn
frying pan
fur coat
garbage truck
gasmask
gas pump
goblet
go-kart
golf ball
golfcart
gondola
gong
gown
grand piano
greenhouse
grille
grocery store
guillotine
hair slide
hair spray
half track
hammer
hamper
hand blower
hand-held computer
handkerchief
hard disc
harmonica
harp
harvester
hatchet
holster
home theater
honeycomb
hook
hoopskirt
horizontal bar
horse cart
hourglass
iPod
iron
jack-o-lantern
jean
jeep
jersey
jigsaw puzzle
jinrikisha
joystick
kimono
knee pad
knot
lab coat
ladle
lampshade
laptop
lawn mower
lens cap
letter opener
library
lifeboat
lighter
limousine
liner
lipstick
Loafer
lotion
loudspeaker
loupe
lumbermill
magnetic compass
mailbag
mailbox
maillot
maillot
manhole cover
maraca
marimba
mask
matchstick
maypole
maze
measuring cup
medicine chest
megalith
microphone
microwave
military uniform
milk can
minibus
miniskirt
minivan
missile
mitten
mixing bowl
mobile home
Model T
modem
monastery
monitor
moped
mortar
mortarboard
mosque
mosquito net
motor scooter
mountain bike
mountain tent
mouse
mousetrap
moving van
muzzle
nail
neck brace
necklace
nipple
notebook
obelisk
oboe
ocarina
odometer
oil filter
organ
oscilloscope
overskirt
oxcart
oxygen mask
packet
paddle
paddlewheel
padlock
paintbrush
pajama
palace
panpipe
paper towel
parachute
parallel bars
park bench
parking meter
passenger car
patio
pay-phone
pedestal
pencil box
pencil sharpener
perfume
Petri dish
photocopier
pick
pickelhaube
picket fence
pickup
pier
piggy bank
pill bottle
pillow
ping-pong ball
pinwheel
pirate
pitcher
plane
planetarium
plastic bag
plate rack
plow
plunger
Polaroid camera
pole
police van
poncho
pool table
pop bottle
pot
potters wheel
power drill
prayer rug
printer
prison
projectile
projector
puck
punching bag
purse
quill
quilt
racer
racket
radiator
radio
radio telescope
rain barrel
recreational vehicle
reel
reflex camera
refrigerator
remote control
restaurant
revolver
rifle
rocking chair
rotisserie
rubber eraser
rugby ball
rule
running shoe
safe
safety pin
saltshaker
sandal
sarong
sax
scabbard
scale
school bus
schooner
scoreboard
screen
screw
screwdriver
seat belt
sewing machine
shield
shoe shop
shoji
shopping basket
shopping cart
shovel
shower cap
shower curtain
ski
ski mask
sleeping bag
slide rule
sliding door
slot
snorkel
snowmobile
snowplow
soap dispenser
soccer ball
sock
solar dish
sombrero
soup bowl
space bar
space heater
space shuttle
spatula
speedboat
spider web
spindle
sports car
spotlight
stage
steam locomotive
steel arch bridge
steel drum
stethoscope
stole
stone wall
stopwatch
stove
strainer
streetcar
stretcher
studio couch
stupa
submarine
suit
sundial
sunglass
sunglasses
sunscreen
suspension bridge
swab
sweatshirt
swimming trunks
swing
switch
syringe
table lamp
tank
tape player
teapot
teddy
television
tennis ball
thatch
theater curtain
thimble
thresher
throne
tile roof
toaster
tobacco shop
toilet seat
torch
totem pole
tow truck
toyshop
tractor
trailer truck
tray
trench coat
tricycle
trimaran
tripod
triumphal arch
trolleybus
trombone
tub
turnstile
typewriter keyboard
umbrella
unicycle
upright
vacuum
vase
vault
velvet
vending machine
vestment
viaduct
violin
volleyball
waffle iron
wall clock
wallet
wardrobe
warplane
washbasin
washer
water bottle
water jug
water tower
whiskey jug
whistle
wig
window screen
window shade
Windsor tie
wine bottle
wing
wok
wooden spoon
wool
worm fence
wreck
yawl
yurt
web site
comic book
crossword puzzle
street sign
traffic light
book jacket
menu
plate
guacamole
consomme
hot pot
trifle
ice cream
ice lolly
French loaf
bagel
pretzel
cheeseburger
hotdog
mashed potato
head cabbage
broccoli
cauliflower
zucchini
spaghetti squash
acorn squash
butternut squash
cucumber
artichoke
bell pepper
cardoon
mushroom
Granny Smith
strawberry
orange
lemon
fig
pineapple
banana
jackfruit
custard apple
pomegranate
hay
carbonara
chocolate sauce
dough
meat loaf
pizza
potpie
burrito
red wine
espresso
cup
eggnog
alp
bubble
cliff
coral reef
geyser
lakeside
promontory
sandbar
seashore
valley
volcano
ballplayer
groom
scuba diver
rapeseed
daisy
yellow ladys slipper
corn
acorn
hip
buckeye
coral fungus
agaric
gyromitra
stinkhorn
earthstar
hen-of-the-woods
bolete
ear
toilet tissue'''
