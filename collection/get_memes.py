from PIL import Image
import requests
from bs4 import BeautifulSoup
import glob
import os
import urllib.request

def get_main_image(meme_url, meme_name):
	response = requests.get(meme_url);
	soup = BeautifulSoup(response.text, 'html.parser')
	image_holder = soup.find("div", {"class" : "details-char-img-inner only-above-768"})
	image_link = image_holder.find("img").get("src")
	print(image_link)
	urllib.request.urlretrieve(image_link, "templates" + "/" + meme_name + ".jpg")

if __name__ == '__main__':
	searches = [("Willy-Wonka", "Condescending Wonka"), ("Futurama-Fry", "Futurama Fry"),
	("The-Most-Interesting-Man-In-The-World", "Most Interesting Man"), ("First-World-Problems", "First World Problems"),
	("Grumpy-Cat", "Grumpy Cat"), ("What-If-I-Told-You", "What If I Told You"),
	("Forever-Alone", "Forever Alone"), ("Conspiracy-Keanu", "Conspiracy Keanu"),
	("Kermit-The-Frog-Drinking-Tea", "Kermit Drinking Tea"), ("Trollface", "Trollface"),
	("Insanity-Wolf", "Insanity Wolf"), ("Yo-Dawg", "Yo Dawg"), ("Disaster-Girl", "Disaster Girl"),
	("Skeptical-3rd-World-Kid", "Skeptical 3rd World Kid"), ("Joseph-Ducreux", "Joseph Ducreux"),
	("Slowpoke", "Slowpoke"), ("Dr-Evil-Meme", "Dr Evil Meme"), ("Joker-Mind-Loss", "Joker Mind Loss"),
	("Stoner-Stanley", "Stoner Stanley"), ("Mr-Bean", "Mr Bean"), ("Good-Guy-Greg", "Good Guy Greg"),
	("Success-Kid", "Success Kid"), ("Bad-Luck-Brian", "Bad Luck Brian"), ("Y-U-No", "Y U No"),
	("One-Does-Not-Simply", "One Does Not Simply"), ("Scumbag-Steve", "Scumbag Steve"),
	("Philosoraptor", "Philosoraptor"), ("Batman-Slap-Robin", "Batman Slap Robin"),
	("Drunk-Baby-1", "Drunk Baby"), ("Correction-Guy", "Correction Guy"), ("Sudden-Realization-Ralph", "Sudden Realization Ralph"),
	("Imagination", "Spongebob Imagination"), ("Southpark-Bad-Time-Meme", "Southpark Bad Time"),
	("Chemistry-Cat", "Chemistry Cat"), ("Captain-Picard", "Captain Picard"), ("So-Doge", "Doge"),
	("awkward-seal", "Awkward Situation Seal"), ("Unpopular-Opinion-Puffin", "Unpopular Opinion Puffin"),
	("Confession-Bear", "Confession Bear"), ("That-Would-Be-Great", "That Would Be Great"),
	]
	
	new_searches = [
	]
	
	site = "https://memegenerator.net/"
	last_bit = "/images/popular/alltime/page/"
	
	opener=urllib.request.build_opener()
	opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
	urllib.request.install_opener(opener)
	
	for search_tuple in searches:
		search = search_tuple[0]
		meme_name = search_tuple[1]
		
		print(meme_name)
		
		meme_url = site + search
		get_main_image(meme_url, meme_name)
		
		
		with open('annotations 2/' + meme_name + '.txt', 'a', encoding="utf-8") as output:
			for page_num in range(70,140):
				url = meme_url + last_bit + str(page_num)
				response = requests.get(url);
				
				soup = BeautifulSoup(response.text, 'html.parser')
				image_links = soup.find_all("div", {"class":"char-img"})
				for image in image_links:
					link_list = image.find("a").get("href").split("/")
					sentence = image.find("img").get("alt")[len(search) + 3:]
					image_id = link_list[2]
					image_presentence = link_list[3]
					output.write(image_id + '\t' + repr(sentence) + '\n')