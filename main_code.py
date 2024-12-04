import art
print(art.logo)

bids={}
biding_finished=False
def who_is_the_winner(bids):
    winner=""
    highest_bid=0
    for bid in bids:
        bidding_amount=bids[bid]
        if bidding_amount>highest_bid:
            highest_bid=bidding_amount
            winner=bid
    print(f"The winner is {winner} with a bid of ${highest_bid}")
while not biding_finished:
    print("Welcome to the secret auction program.")
    name=input("what is your name ?: ")
    price=int(input("what is your bid ?: $"))
    bids[name]=price

    more_entries=input("Are there any other bidders? Type 'yes 'or 'no'").lower()
    if more_entries=="yes":
        print("\n"*20)
    elif more_entries=="no":
        biding_finished=True
        who_is_the_winner(bids)





