import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class CifarLoader(object):
	"""docstring for CifarLoader"""
	def __init__(self, args):
		super(CifarLoader, self).__init__()
		transform = transforms.Compose(
		    [
		     # TODO: Add data augmentations here
		     transforms.ToTensor(),
		     #transforms.ColorJitter(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		     ])

		transform_test = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
		])

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		                                        download=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		                                       download=True, transform=transform_test) 
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
		                                         shuffle=False, num_workers=2)

		self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		
class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.classes = ("Tundra Swan","Yellow-billed Magpie","Bay-breasted Warbler (Female, Nonbreeding male, Immature)","Common Goldeneye (Female/Eclipse male)","Hermit Thrush","Green-winged Teal (Male)","Magnolia Warbler (Breeding male)","Hoary Redpoll","Wilson's Snipe","Lesser Yellowlegs","Verdin","Dark-eyed Junco (Pink-sided)","Western Tanager (Breeding Male)","Brant","Snow Bunting (Nonbreeding)","Common Goldeneye (Breeding male)","American Wigeon (Breeding male)","Harris's Sparrow (Immature)","Eurasian Collared-Dove","Red-necked Grebe (Nonbreeding/juvenile)","Roseate Spoonbill","Brewer's Blackbird (Female/Juvenile)","Snow Goose (White morph)","Mountain Chickadee","Summer Tanager (Adult Male)","Inca Dove","Costa's Hummingbird (Adult Male)","Bell's Vireo","Black-crowned Night-Heron (Immature)","Fox Sparrow (Red)","Hooded Merganser (Breeding male)","Pacific Loon (Breeding)","Palm Warbler","Common Tern","Bushtit","Horned Grebe (Nonbreeding/juvenile)","Cackling Goose","Black-chinned Hummingbird (Adult Male)","Green-winged Teal  (Female/juvenile)","Herring Gull (Adult)","Mallard (Female/Eclipse male)","Sharp-shinned Hawk (Immature)","Killdeer","Nashville Warbler","Eastern Phoebe","Golden-crowned Kinglet","Green-tailed Towhee","Orchard Oriole (Adult Male)","Black-bellied Plover (Nonbreeding/juvenile)","Veery","Scissor-tailed Flycatcher","Bullock's Oriole (Female/Immature male)","Northern Harrier (Adult male)","Mew Gull","Cassin's Finch (Adult Male)","Sanderling (Nonbreeding/juvenile)","California Thrasher","Redhead (Female/Eclipse male)","Broad-tailed Hummingbird (Adult Male)","Lark Bunting (Breeding male)","White-breasted Nuthatch","Phainopepla (Female/juvenile)","Blue-winged Teal  (Male)","Brown Creeper","Red-headed Woodpecker (Immature)","Greater Yellowlegs","Cinnamon Teal (Female/juvenile)","Mute Swan","Black Scoter (Female/juvenile)","Common Loon (Nonbreeding/juvenile)","Eastern Towhee","Black-crowned Night-Heron (Adult)","Laughing Gull (Nonbreeding/Immature)","Fish Crow","Cooper's Hawk (Adult)","Red-naped Sapsucker","Forster's Tern","Vermilion Flycatcher (Adult male)","Lesser Scaup (Breeding male)","Laughing Gull (Breeding)","Spotted Towhee","Yellow-rumped Warbler (Breeding Audubon's)","Chimney Swift","Magnolia Warbler (Female/immature male)","Common Eider (Immature/Eclipse male)","Eastern Bluebird","Clay-colored Sparrow","Ruby-throated Hummingbird (Female, immature)","Hooded Merganser (Female/immature male)","Sanderling (Breeding)","Field Sparrow","Lesser Goldfinch (Adult Male)","American Redstart (Female/juvenile)","Red-winged Blackbird (Male)","American Goldfinch (Female/Nonbreeding Male)","Mississippi Kite","Swainson's Hawk (Dark morph )","Surf Scoter (Male)","White-throated Sparrow (White-striped)","Black-billed Cuckoo","Rough-legged Hawk (Light morph)","Abert's Towhee","Northern Rough-winged Swallow","Red-shouldered Hawk (Immature)","Purple Martin (Female/juvenile)","Blue Jay","Varied Thrush","Clark's Grebe","Reddish Egret (White morph)","Spotted Sandpiper (Breeding)","Western Grebe","Little Blue Heron (Immature)","Broad-billed Hummingbird (Adult Male)","Say's Phoebe","White-crowned Sparrow (Immature)","White Ibis (Immature)","Western Gull (Adult)","California Quail (Male)","White-crowned Sparrow (Adult)","Ring-necked Duck (Breeding male)","Bewick's Wren","Northern Pintail (Breeding male)","American Pipit","Blackpoll Warbler (Female/juvenile)","Belted Kingfisher","Winter Wren","American Black Duck","Barrow's Goldeneye (Female/Eclipse male)","Fox Sparrow (Sooty)","Purple Martin (Adult male)","Wood Duck (Female/Eclipse male)","Blue Grosbeak (Female/juvenile)","White-tailed Kite","American Coot","Snow Goose (Blue morph)","Great Black-backed Gull (Adult)","White-winged Crossbill (Female/juvenile)","Anna's Hummingbird (Adult Male)","Ring-necked Pheasant (Female/juvenile)","Dark-eyed Junco (Oregon)","Mallard (Breeding male)","Ruby-crowned Kinglet","Brown-headed Cowbird (Male)","Common Yellowthroat (Female/immature male)","Gray Jay","House Wren","Trumpeter Swan","Mourning Dove","Summer Tanager (Immature Male)","Common Loon (Breeding)","Black-legged Kittiwake (Immature)","Cassin's Finch (Female/immature)","Tennessee Warbler","Cactus Wren","Evening Grosbeak (Female/Juvenile)","Red-necked Grebe (Breeding)","Blue-headed Vireo","Whimbrel","MacGillivray's Warbler","Reddish Egret (Dark morph)","Willet","Cassin's Kingbird","Lesser Scaup (Female/Eclipse male)","Northern Gannet (Adult, Subadult)","Osprey","Bonaparte's Gull","Black-throated Blue Warbler (Adult Male)","Pacific Loon (Nonbreeding/juvenile)","Canvasback (Breeding male)","Indigo Bunting (Female/juvenile)","Pine Grosbeak (Adult Male)","Red-throated Loon (Nonbreeding/juvenile)","Marbled Godwit","Yellow-throated Warbler","Common Raven","Steller's Jay","Black-billed Magpie","House Finch (Female/immature)","Spotted Sandpiper (Nonbreeding/juvenile)","Pygmy Nuthatch","Brandt's Cormorant","Black-crested Titmouse","Long-tailed Duck (Winter male)","Swallow-tailed Kite","Brown Thrasher","Black-chinned Hummingbird (Female, immature)","Black-throated Gray Warbler","Snowy Owl","White-winged Dove","Surf Scoter (Female/immature)","American Tree Sparrow","Lesser Goldfinch (Female/juvenile)","Northern Flicker (Yellow-shafted)","Lark Sparrow","Rose-breasted Grosbeak (Adult Male)","Northern Gannet (Immature/Juvenile)","Yellow-headed Blackbird (Female/Immature Male)","Prairie Falcon","Dickcissel","Warbling Vireo","Wood Thrush","Harris's Hawk","Pacific Wren","Eared Grebe (Nonbreeding/juvenile)","Least Flycatcher","Baltimore Oriole (Female/Immature male)","Harris's Sparrow (Adult)","Wood Stork","Hutton's Vireo","Glaucous-winged Gull (Immature)","Peregrine Falcon (Immature)","Black-necked Stilt","Glaucous-winged Gull (Adult)","Pigeon Guillemot (Breeding)","Boreal Chickadee","Red-breasted Sapsucker","Yellow-rumped Warbler (Breeding Myrtle)","Blackburnian Warbler","Red-breasted Merganser (Breeding male)","Gadwall (Breeding male)","Barred Owl","Common Nighthawk","Painted Bunting (Female/juvenile)","Costa's Hummingbird (Female, immature)","Townsend's Solitaire","Common Yellowthroat (Adult Male)","Bald Eagle (Immature, juvenile)","Eastern Screech-Owl","Common Grackle","Pacific-slope Flycatcher","Florida Scrub-Jay","Black-and-white Warbler","Cordilleran Flycatcher","Little Blue Heron (Adult)","California Gull (Adult)","Evening Grosbeak (Adult Male)","Monk Parakeet","Great Crested Flycatcher","Wilson's Phalarope (Breeding)","Broad-winged Hawk (Adult)","Burrowing Owl","Pyrrhuloxia","Nuttall's Woodpecker","Herring Gull (Immature)","Bay-breasted Warbler (Breeding male)","Surfbird","Cape May Warbler","Dark-eyed Junco (Red-backed/Gray-headed)","Glossy Ibis","Band-tailed Pigeon","Gila Woodpecker","Western Sandpiper","Pine Siskin","Fox Sparrow (Thick-billed/Slate-colored)","Common Gallinule (Immature)","Vesper Sparrow","White-eyed Vireo","Indigo Bunting (Adult Male)","Barn Owl","Great-tailed Grackle","Northern Harrier (Female, immature)","Clark's Nutcracker","American Crow","Yellow-rumped Warbler (Winter/juvenile Myrtle)","Long-billed Curlew","Royal Tern","Horned Lark","Common Gallinule (Adult)","Swamp Sparrow","White-throated Swift","Dunlin (Nonbreeding/juvenile)","House Sparrow (Male)","Caspian Tern","Black-bellied Whistling-Duck","American Woodcock","Wild Turkey","Hooded Oriole (Adult male)","Blue Grosbeak (Adult Male)","Northern Parula","Common Ground-Dove","Cassin's Vireo","Greater Roadrunner","Golden-crowned Sparrow (Adult)","Common Eider (Female/juvenile)","Canyon Wren","Bald Eagle (Adult, subadult)","Orange-crowned Warbler","Red Crossbill (Female/juvenile)","Calliope Hummingbird (Female, immature)","Harlequin Duck (Male)","Pine Grosbeak (Female/juvenile)","Western Bluebird","Golden-fronted Woodpecker","American Kestrel (Female, immature)","Broad-tailed Hummingbird (Female, immature)","Purple Gallinule (Adult)","Ring-billed Gull (Adult)","Chihuahuan Raven","Double-crested Cormorant (Immature)","Dunlin (Breeding)","Purple Gallinule (Immature)","Red-breasted Nuthatch","Red-shouldered Hawk (Adult )","Northern Shrike","Black Guillemot (Nonbreeding, juvenile)","Swainson's Thrush","Blackpoll Warbler (Breeding male)","Red-headed Woodpecker (Adult)","Great Black-backed Gull (Immature)","Lazuli Bunting (Adult Male)","Orchard Oriole (Female/Juvenile)","European Starling (Breeding Adult)","Chestnut-sided Warbler (Female/immature male)","Baltimore Oriole (Adult male)","Yellow-throated Vireo","Painted Bunting (Adult Male)","American Redstart (Adult Male)","American Robin (Juvenile)","Purple Finch (Adult Male)","Snow Bunting (Breeding adult)","Chipping Sparrow (Breeding)","Red-tailed Hawk (Dark morph)","Violet-green Swallow","Rock Pigeon","Yellow-crowned Night-Heron (Immature)","American Dipper","European Starling (Nonbreeding Adult)","Black Tern","Yellow Warbler","American Oystercatcher","Allen's Hummingbird (Female, immature)","Ring-billed Gull (Immature)","White-winged Scoter (Female/juvenile)","Gadwall (Female/Eclipse male)","Black-capped Chickadee","California Towhee","Western Kingbird","Rufous Hummingbird (Adult Male)","Black-bellied Plover (Breeding)","Tricolored Heron","Northern Waterthrush","Western Screech-Owl","Phainopepla (Male)","Peregrine Falcon (Adult)","Double-crested Cormorant (Adult)","Redhead (Breeding male)","Red-bellied Woodpecker","American Avocet","Hooded Oriole (Female/Immature male)","Louisiana Waterthrush","Yellow-breasted Chat","American Robin (Adult)","Semipalmated Sandpiper","Northern Flicker (Red-shafted)","Black-headed Grosbeak (Adult Male)","Purple Finch (Female/immature)","Turkey Vulture","Harlequin Duck (Female/juvenile)","Northern Cardinal (Female/Juvenile)","Bufflehead (Female/immature male)","California Quail (Female/juvenile)","Bobolink (Breeding male)","Red-tailed Hawk (Light morph immature)","Chipping Sparrow (Immature/nonbreeding adult)","Greater Scaup (Breeding male)","Dark-eyed Junco (Slate-colored)","Canyon Towhee","Lark Bunting (Female/Nonbreeding male)","Great Horned Owl","Wood Duck (Breeding male)","Rusty Blackbird","Brown Pelican","Brown-headed Cowbird (Female/Juvenile)","American Wigeon (Female/Eclipse male)","Oak Titmouse","White-winged Scoter (Male)","Marsh Wren","Northern Pintail (Female/Eclipse male)","Horned Grebe (Breeding)","Chestnut-backed Chickadee","Ring-necked Pheasant (Male)","Common Redpoll","White-winged Crossbill (Adult Male)","Brewer's Blackbird (Male)","Barn Swallow","Juniper Titmouse","Snowy Egret","Northwestern Crow","Bobolink (Female/juvenile/nonbreeding male)","Heermann's Gull (Immature)","Bohemian Waxwing","Common Eider (Adult male)","Blue-gray Gnatcatcher","Tufted Titmouse","House Sparrow (Female/Juvenile)","Black Scoter (Male)","Neotropic Cormorant","Mountain Bluebird","Sharp-shinned Hawk (Adult )","Ruddy Duck (Female/juvenile)","Cooper's Hawk (Immature)","Red-eyed Vireo","Black Rosy-Finch","Carolina Chickadee","Blue-winged Warbler","White-faced Ibis","Lazuli Bunting (Female/juvenile)","Western Meadowlark","Great Blue Heron","Downy Woodpecker","Blue-winged Teal  (Female/juvenile)","Yellow-headed Blackbird (Adult Male)","Wilson's Phalarope (Nonbreeding, juvenile)","Loggerhead Shrike","Cinnamon Teal (Male)","Vaux's Swift","Golden Eagle (Immature)","Long-tailed Duck (Female/juvenile)","Ruffed Grouse","Scarlet Tanager (Female/Nonbreeding Male)","Bridled Titmouse","Mourning Warbler","Ross's Goose","Ruddy Duck (Breeding male)","Red Crossbill (Adult Male)","Gray-crowned Rosy-Finch","Yellow-billed Cuckoo","Savannah Sparrow","Eastern Meadowlark","Bufflehead (Breeding male)","Black-headed Grosbeak (Female/immature male)","Black-throated Blue Warbler (Female/Immature male)","Sandhill Crane","Wrentit","Black-throated Green Warbler","Rufous-crowned Sparrow","Green Heron","Broad-billed Hummingbird (Female, immature)","Black Skimmer","Prothonotary Warbler","Black-tailed Gnatcatcher","Northern Pygmy-Owl","Black Oystercatcher","California Gull (Immature)","Black-legged Kittiwake (Adult)","Long-tailed Duck (Summer male)","Red-winged Blackbird (Female/juvenile)","Black Turnstone","Swainson's Hawk (Light morph )","Short-billed Dowitcher","Prairie Warbler","White-throated Sparrow (Tan-striped/immature)","Pine Warbler","Wilson's Warbler","Pelagic Cormorant","Common Merganser (Female/immature male)","Ladder-backed Woodpecker","Least Sandpiper","White Ibis (Adult)","Northern Saw-whet Owl","Red-throated Loon (Breeding)","Ruddy Turnstone","Scarlet Tanager (Breeding Male)","Brewer's Sparrow","Hairy Woodpecker","Yellow-rumped Warbler (Winter/juvenile Audubon's)","Ring-necked Duck (Female/Eclipse male)","Vermilion Flycatcher (Female, immature)","Red-breasted Merganser (Female/immature male)","Western Scrub-Jay","Boat-tailed Grackle","Rose-breasted Grosbeak (Female/immature male)","Song Sparrow","Great Cormorant (Immature)","Orchard Oriole (Immature Male)","Bronzed Cowbird","Great Cormorant (Adult)","Calliope Hummingbird (Adult Male)","Greater White-fronted Goose","Allen's Hummingbird (Adult Male)","Gambel's Quail (Female/juvenile)","Eastern Wood-Pewee","Western Tanager (Female/Nonbreeding Male)","Canada Warbler","Golden-crowned Sparrow (Immature)","Northern Shoveler (Female/Eclipse male)","Cattle Egret","Western Gull (Immature)","Scaled Quail","Tree Swallow","Gambel's Quail (Male)","Plumbeous Vireo","Crested Caracara","Dark-eyed Junco (White-winged)","Red-tailed Hawk (Light morph adult)","Carolina Wren","Anna's Hummingbird (Female, immature)","Pileated Woodpecker","Cave Swallow","Cedar Waxwing","Broad-winged Hawk (Immature)","Rough-legged Hawk (Dark morph)","Barrow's Goldeneye (Breeding male)","American Kestrel (Adult male)","Black Vulture","American Goldfinch (Breeding Male)","Ruby-throated Hummingbird (Adult Male)","Brown-headed Nuthatch","Swainson's Hawk (Immature)","Great Egret","Solitary Sandpiper","Bullock's Oriole (Adult male)","Townsend's Warbler","Pigeon Guillemot (Nonbreeding, juvenile)","Curve-billed Thrasher","Cliff Swallow","Anhinga","Chestnut-sided Warbler (Breeding male)","Yellow-bellied Sapsucker","Greater Scaup (Female/Eclipse male)","Yellow-crowned Night-Heron (Adult)","Western Wood-Pewee","House Finch (Adult Male)","Hermit Warbler","Ovenbird","Black Phoebe","Bank Swallow","Northern Shoveler (Breeding male)","Common Merganser (Breeding male)","American White Pelican","Ruddy Duck (Winter male)","Black Guillemot (Breeding)","Mexican Jay","Ash-throated Flycatcher","Lincoln's Sparrow","Canvasback (Female/Eclipse male)","Gray Catbird","Semipalmated Plover","Northern Mockingbird","European Starling (Juvenile)","Pied-billed Grebe","Merlin","Rufous Hummingbird (Female, immature)","Eared Grebe (Breeding)","Heermann's Gull (Adult)","Brown-capped Rosy-Finch","Eastern Kingbird","Golden Eagle (Adult)","Hooded Warbler","Acorn Woodpecker","Northern Cardinal (Adult Male)","Canada Goose","Northern Bobwhite","Summer Tanager (Female)","Mottled Duck")

    def __getitem__(self, index):
        # Get image name from the pandas df
        print(self.image_arr[index])
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        # print("EH")
		cNorm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		img_as_tensor = cNorm(img_as_tensor)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len












