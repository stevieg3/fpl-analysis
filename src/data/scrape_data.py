import pandas as pd
import numpy as np
from selenium import webdriver
import time
from selenium.common.exceptions import NoSuchElementException
import logging
import pickle

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ScrapeData:
    def __init__(self, number_of_pages, max_players_per_page):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("detach", True)  # keep driver open
        chrome_options.headless = True
        self.driver = webdriver.Chrome(
            "/Users/stevengeorge/Documents/Python/Chrome Driver/chromedriver",
            options=chrome_options,
        )
        self.driver.implicitly_wait(10)
        self.driver.get("https://fantasy.premierleague.com/a/statistics/total_points")

        self.number_of_pages = number_of_pages
        self.max_players_per_page = max_players_per_page

    def _get_player_data(self, player_num):
        """
        Scrape tabular data for a single player

        :param player_num: Position of player on page. Any value from 1 to self.max_players_per_page
        :return: Dictionary of DataFrames containing current and previous season data
        """
        self.driver.find_element_by_xpath(
            '//*[@id="ismr-main"]/div/div[3]/table/tbody/tr[{}]/td[2]/div/div[2]/a'.format(
                str(player_num)
            )
        ).click()
        time.sleep(1)  # Get raw data as string

        tbl = self.driver.find_element_by_xpath(
            '//*[@id="ism-eiw-history"]'
        ).get_attribute("outerHTML")

        stats = pd.read_html(
            tbl
        )  # Create list of DataFrames (containing current and previous season if available)

        current_season = stats[0]  # Select first table (current season)
        try:
            previous_season = stats[1]
        except IndexError:
            previous_season = "First season in PL"

        self.driver.find_element_by_xpath(
            '//*[@id="ismr-element"]/div/div[1]/a'
        ).click()  # close box
        time.sleep(1)
        full_data = {
            "current_season": current_season,
            "previous_season": previous_season,
        }

        return full_data

    def _get_player_details(self, player_num):
        """
        Get player name, team and position

        :param player_num: Position of player on page. Any value from 1 to self.max_players_per_page
        :return: Dictionary containing player descriptions
        """
        name = self.driver.find_element_by_xpath(
            '//*[@id="ismr-main"]/div/div[3]/table/tbody/tr[{}]/td[2]/div/div[2]/a'.format(
                str(player_num)
            )
        ).text
        team = self.driver.find_element_by_xpath(
            '//*[@id="ismr-main"]/div/div[3]/table/tbody/tr[{}]/td[2]/div/div[2]/span[1]'.format(
                str(player_num)
            )
        ).text
        pos = self.driver.find_element_by_xpath(
            '//*[@id="ismr-main"]/div/div[3]/table/tbody/tr[{}]/td[2]/div/div[2]/span[2]'.format(
                str(player_num)
            )
        ).text

        full_details = {"name": name, "team": team, "position": pos}

        return full_details

    def _scrape_page(self):
        """
        Scrapes current and previous season data for every player on a given page

        :return: Dictionary containing current and previous season data
        """
        player_data_curr = []
        player_data_prev = {}
        try:
            for player in np.arange(1, self.max_players_per_page + 1):
                all_season_data = self._get_player_data(player_num=player)
                details = self._get_player_details(player_num=player)

                time.sleep(2)

                current_season_df = all_season_data["current_season"]
                current_season_df["Name"] = details["name"]
                current_season_df["Team"] = details["team"]
                current_season_df["Position"] = details["position"]
                player_data_curr.append(current_season_df)

                previous_season_data = all_season_data["previous_season"]
                player_data_prev[
                    (details["name"], details["team"], details["position"])
                ] = previous_season_data

                logging.info(f'{details["name"]} completed')

        except NoSuchElementException:
            pass

        page_data = {
            "current_season": player_data_curr,
            "previous_season": player_data_prev,
        }

        return page_data

    def scrape_all(self):
        """
        Cycles through every page on site and scrapes all player data. Saves previous season data as pickle and current
        season data to csv

        :return: None
        """
        current_season_list = []
        previous_season_list = []
        for page_number in np.arange(1, self.number_of_pages + 1):
            if page_number == 1:
                all_page_data = self._scrape_page()
                current_season = all_page_data["current_season"]
                previous_season = all_page_data["previous_season"]
                current_season_list.append(current_season)
                previous_season_list.append(previous_season)
                self.driver.find_element_by_xpath(
                    '//*[@id="ismr-main"]/div/div[4]/a[1]'
                ).click()  # go to next page
            elif (page_number >= 2) & (page_number < self.number_of_pages):
                all_page_data = self._scrape_page()
                current_season = all_page_data["current_season"]
                previous_season = all_page_data["previous_season"]
                current_season_list.append(current_season)
                previous_season_list.append(previous_season)
                self.driver.find_element_by_xpath(
                    '//*[@id="ismr-main"]/div/div[4]/a[3]'
                ).click()  # go to next page
            else:
                all_page_data = self._scrape_page()
                current_season = all_page_data["current_season"]
                previous_season = all_page_data["previous_season"]
                current_season_list.append(current_season)
                previous_season_list.append(previous_season)

            logging.info(f"Page {page_number} completed")

        f = open("../data/2018_19_previous_season_data.pickle", "wb")
        pickle.dump(previous_season_list, f)

        unpacked = [
            pd.concat(x) for x in current_season_list
        ]  # Each item in current_season_list is a list of 30 dfs
        combined = pd.concat(unpacked)
        combined = combined[combined["GW  Gameweek"] != "Totals"]

        combined.to_csv("../data/2018_19_current_season_data.csv", index=False)

        self.driver.close()


def main():
    fpl_scraper = ScrapeData(number_of_pages=18, max_players_per_page=30)
    fpl_scraper.scrape_all()


if __name__ == "__main__":
    main()
