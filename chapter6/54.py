from gensim.models import KeyedVectors

# Google Newsで学習済みの単語ベクトルファイル
# 事前に GoogleNews-vectors-negative300.bin.gz をダウンロードして、
# このPythonファイルと同じフォルダに置いておく
file_path = "GoogleNews-vectors-negative300.bin.gz"

# 単語アナロジーの評価データ
analogy_file = "questions-words.txt"

# 54番の結果を保存するファイル
output_file = "answer_54.txt"

# word2vec形式の学習済み単語ベクトルを読み込む
# binary=True は、このファイルがバイナリ形式で保存されているため必要
model = KeyedVectors.load_word2vec_format(file_path, binary=True)

# 結果を書き込むファイルを開く
with (
    open(analogy_file, "r", encoding="UTF-8") as f,
    open(output_file, "w", encoding="UTF-8") as out,
):

    # 現在読んでいるセクションがcapital-common-countriesかどうか
    in_target_section = False
    # ファイルを1行ずつ読む
    for line in f:
        # 行末の改行を削除する
        line = line.strip()

        # セクション名の行は":"で始まる
        if line.startswith(":"):
            # capital-common-countries セクションに入ったかを判定する
            if line == ": capital-common-countries":
                in_target_section = True
            else:
                in_target_section = False

            # セクション名の行は問題ではないので、次の行へ進む
            continue

        # capital-common-countries 以外のセクションは処理しない
        # notをつけているから、in_target_sectionがFalseなら
        # Trueとなり（反対になり）、continueで飛ばされる
        if not in_target_section:
            continue

        # 1行を4つの単語に分ける
        # 例: Athens Greece Baghdad Iraq
        word1, word2, word3, word4 = line.split()

        # vec(2列目) - vec(1列目) + vec(3列目) に最も近い単語を求める
        # positive は足す単語、negative は引く単語
        result_word, similarity = model.most_similar(
            positive=[word2, word3], negative=[word1], topn=1
        )[0]

        # 元の4単語に、予測した単語と類似度を追加して保存する
        out.write(f"{word1} {word2} {word3} {word4} {result_word} {similarity}\n")

# 出力
# Athens Greece Baghdad Iraq Iraqi 0.635187029838562
# Athens Greece Bangkok Thailand Thailand 0.7137669920921326
# Athens Greece Beijing China China 0.7235777378082275
# Athens Greece Berlin Germany Germany 0.673462450504303
# Athens Greece Bern Switzerland Switzerland 0.4919748306274414
# Athens Greece Cairo Egypt Egypt 0.7527808547019958
# Athens Greece Canberra Australia Australia 0.5837326049804688
# Athens Greece Hanoi Vietnam Viet_Nam 0.6276342272758484
# Athens Greece Havana Cuba Cuba 0.6460990905761719
# Athens Greece Helsinki Finland Finland 0.6899983286857605
# Athens Greece Islamabad Pakistan Pakistan 0.7233325839042664
# Athens Greece Kabul Afghanistan Afghan 0.6160915493965149
# Athens Greece London England Britain 0.5646187663078308
# Athens Greece Madrid Spain Spain 0.7036614418029785
# Athens Greece Moscow Russia Russia 0.7382972240447998
# Athens Greece Oslo Norway Norway 0.6470743417739868
# Athens Greece Ottawa Canada Canada 0.5912168622016907
# Athens Greece Paris France France 0.672462522983551
# Athens Greece Rome Italy Italy 0.6826189756393433
# Athens Greece Stockholm Sweden Sweden 0.6757242679595947
# Athens Greece Tehran Iran Iran 0.7671369314193726
# Athens Greece Tokyo Japan Japan 0.725609540939331
# Baghdad Iraq Bangkok Thailand Thailand 0.6745353937149048
# Baghdad Iraq Beijing China China 0.6915647983551025
# Baghdad Iraq Berlin Germany Germany 0.6057409644126892
# Baghdad Iraq Bern Switzerland coach_Bobby_Curlings 0.4347156584262848
# Baghdad Iraq Cairo Egypt Egypt 0.684848964214325
# Baghdad Iraq Canberra Australia Mr_Rudd 0.6186109185218811
# Baghdad Iraq Hanoi Vietnam Vietnam 0.7493553757667542
# Baghdad Iraq Havana Cuba Cuba 0.7133256196975708
# Baghdad Iraq Helsinki Finland Finland 0.597256064414978
# Baghdad Iraq Islamabad Pakistan Pakistan 0.7466937303543091
# Baghdad Iraq Kabul Afghanistan Afghanistan 0.8359358906745911
# Baghdad Iraq London England Britain 0.5853632092475891
# Baghdad Iraq Madrid Spain Spain 0.6144677996635437
# Baghdad Iraq Moscow Russia Russia 0.7236211895942688
# Baghdad Iraq Oslo Norway Norway 0.5987578630447388
# Baghdad Iraq Ottawa Canada Prime_Minister_Jean_Chrétien 0.5704856514930725
# Baghdad Iraq Paris France France 0.6203721761703491
# Baghdad Iraq Rome Italy Italy 0.5362657904624939
# Baghdad Iraq Stockholm Sweden Sweden 0.5907801985740662
# Baghdad Iraq Tehran Iran Iran 0.8114985227584839
# Baghdad Iraq Tokyo Japan Japan 0.7056294679641724
# Baghdad Iraq Athens Greece Greece 0.5793530344963074
# Bangkok Thailand Beijing China China 0.8263984322547913
# Bangkok Thailand Berlin Germany Germany 0.729519784450531
# Bangkok Thailand Bern Switzerland Switzerland 0.5047770738601685
# Bangkok Thailand Cairo Egypt Egypt 0.8118714094161987
# Bangkok Thailand Canberra Australia Australia 0.7154911756515503
# Bangkok Thailand Hanoi Vietnam Viet_Nam 0.7700591683387756
# Bangkok Thailand Havana Cuba Cuba 0.8007405996322632
# Bangkok Thailand Helsinki Finland Finland 0.7515275478363037
# Bangkok Thailand Islamabad Pakistan Pakistan 0.7976752519607544
# Bangkok Thailand Kabul Afghanistan Afghanistan 0.6666890382766724
# Bangkok Thailand London England Britain 0.6576253175735474
# Bangkok Thailand Madrid Spain Spain 0.7655300498008728
# Bangkok Thailand Moscow Russia Russia 0.8613656163215637
# Bangkok Thailand Oslo Norway Norway 0.7142738103866577
# Bangkok Thailand Ottawa Canada Canada 0.6684818863868713
# Bangkok Thailand Paris France France 0.7388159036636353
# Bangkok Thailand Rome Italy Italy 0.6863129734992981
# Bangkok Thailand Stockholm Sweden Sweden 0.7587180137634277
# Bangkok Thailand Tehran Iran Iran 0.8363438248634338
# Bangkok Thailand Tokyo Japan Japan 0.8282637596130371
# Bangkok Thailand Athens Greece Greece 0.6765021681785583
# Bangkok Thailand Baghdad Iraq Iraqi 0.6885136365890503
# Beijing China Berlin Germany Germany 0.7232259511947632
# Beijing China Bern Switzerland Bern_NC 0.4655817151069641
# Beijing China Cairo Egypt Egypt 0.7653075456619263
# Beijing China Canberra Australia Australia 0.7234393954277039
# Beijing China Hanoi Vietnam Viet_Nam 0.7793148159980774
# Beijing China Havana Cuba Cuba 0.7189655900001526
# Beijing China Helsinki Finland Finland 0.7141392827033997
# Beijing China Islamabad Pakistan Pakistan 0.806240439414978
# Beijing China Kabul Afghanistan Afghan 0.6770888566970825
# Beijing China London England UK 0.6304796934127808
# Beijing China Madrid Spain Spain 0.7320306301116943
# Beijing China Moscow Russia Russia 0.8579801321029663
# Beijing China Oslo Norway Norway 0.6628976464271545
# Beijing China Ottawa Canada Canada 0.6537172198295593
# Beijing China Paris France France 0.7109121084213257
# Beijing China Rome Italy Italy 0.6637485027313232
# Beijing China Stockholm Sweden Sweden 0.7026645541191101
# Beijing China Tehran Iran Iran 0.8262744545936584
# Beijing China Tokyo Japan Japan 0.823669970035553
# Beijing China Athens Greece Greece 0.6515147686004639
# Beijing China Baghdad Iraq Mosul 0.7041884660720825
# Beijing China Bangkok Thailand Thailand 0.8047245740890503
# Berlin Germany Bern Switzerland Switzerland 0.6095317602157593
# Berlin Germany Cairo Egypt Egypt 0.6912307739257812
# Berlin Germany Canberra Australia Australia 0.7485095858573914
# Berlin Germany Hanoi Vietnam Viet_Nam 0.6875088810920715
# Berlin Germany Havana Cuba Cuba 0.6961181163787842
# Berlin Germany Helsinki Finland Finland 0.7654491662979126
# Berlin Germany Islamabad Pakistan Pakistan 0.7597626447677612
# Berlin Germany Kabul Afghanistan Afghan 0.6450541019439697
# Berlin Germany London England UK 0.6597432494163513
# Berlin Germany Madrid Spain Spain 0.7768210172653198
# Berlin Germany Moscow Russia Russia 0.810738205909729
# Berlin Germany Oslo Norway Norway 0.7291737198829651
# Berlin Germany Ottawa Canada Canada 0.6773227453231812
# Berlin Germany Paris France France 0.7884094715118408
# Berlin Germany Rome Italy Italy 0.7509608268737793
# Berlin Germany Stockholm Sweden Sweden 0.8022657632827759
# Berlin Germany Tehran Iran Iran 0.7747757434844971
# Berlin Germany Tokyo Japan Japan 0.7969786524772644
# Berlin Germany Athens Greece Greece 0.6390470862388611
# Berlin Germany Baghdad Iraq Mosul 0.686406135559082
# Berlin Germany Bangkok Thailand Thailand 0.768083393573761
# Berlin Germany Beijing China China 0.7755852937698364
# Bern Switzerland Cairo Egypt Egypt 0.5989274978637695
# Bern Switzerland Canberra Australia Australia 0.6860505938529968
# Bern Switzerland Hanoi Vietnam Viet_Nam 0.5609369874000549
# Bern Switzerland Havana Cuba Cuba 0.6295959949493408
# Bern Switzerland Helsinki Finland Finland 0.6637657880783081
# Bern Switzerland Islamabad Pakistan Pakistan 0.6708025336265564
# Bern Switzerland Kabul Afghanistan Afghanistan 0.5338183641433716
# Bern Switzerland London England EURASIAN_NATURAL_RESOURCES_CORP. 0.5463826060295105
# Bern Switzerland Madrid Spain Spain 0.6423718333244324
# Bern Switzerland Moscow Russia Russia 0.6487589478492737
# Bern Switzerland Oslo Norway Norway 0.6448377966880798
# Bern Switzerland Ottawa Canada Canada 0.6468111872673035
# Bern Switzerland Paris France France 0.6129554510116577
# Bern Switzerland Rome Italy Italy 0.6486325860023499
# Bern Switzerland Stockholm Sweden Sweden 0.6849664449691772
# Bern Switzerland Tehran Iran Iran 0.6813834309577942
# Bern Switzerland Tokyo Japan Japan 0.6434274911880493
# Bern Switzerland Athens Greece Greece 0.5800100564956665
# Bern Switzerland Baghdad Iraq Iraqi 0.5730240345001221
# Bern Switzerland Bangkok Thailand Thailand 0.6617313623428345
# Bern Switzerland Beijing China China 0.6225298643112183
# Bern Switzerland Berlin Germany Germany 0.7138351202011108
# Cairo Egypt Canberra Australia Australia 0.6811555027961731
# Cairo Egypt Hanoi Vietnam Viet_Nam 0.7589390873908997
# Cairo Egypt Havana Cuba Cuba 0.763901948928833
# Cairo Egypt Helsinki Finland Finland 0.7256740927696228
# Cairo Egypt Islamabad Pakistan Pakistan 0.8155921101570129
# Cairo Egypt Kabul Afghanistan Afghan 0.6829640865325928
# Cairo Egypt London England Britain 0.6454266905784607
# Cairo Egypt Madrid Spain Spain 0.7469130754470825
# Cairo Egypt Moscow Russia Russia 0.842657208442688
# Cairo Egypt Oslo Norway Norway 0.6975642442703247
# Cairo Egypt Ottawa Canada Canada 0.6659148335456848
# Cairo Egypt Paris France France 0.7114198803901672
# Cairo Egypt Rome Italy Italy 0.6812829971313477
# Cairo Egypt Stockholm Sweden Sweden 0.7365163564682007
# Cairo Egypt Tehran Iran Iran 0.8531844019889832
# Cairo Egypt Tokyo Japan Japan 0.7753520011901855
# Cairo Egypt Athens Greece Greece 0.7355356216430664
# Cairo Egypt Baghdad Iraq Iraqi 0.7208045125007629
# Cairo Egypt Bangkok Thailand Thailand 0.8275743126869202
# Cairo Egypt Beijing China China 0.8066044449806213
# Cairo Egypt Berlin Germany Germany 0.7016906142234802
# Cairo Egypt Bern Switzerland Switzerland 0.49304065108299255
# Canberra Australia Hanoi Vietnam Viet_Nam 0.6714490652084351
# Canberra Australia Havana Cuba Cuba 0.6964417695999146
# Canberra Australia Helsinki Finland Finland 0.6906550526618958
# Canberra Australia Islamabad Pakistan Pakistan 0.7743954658508301
# Canberra Australia Kabul Afghanistan Afghans 0.585472047328949
# Canberra Australia London England UK 0.6224532723426819
# Canberra Australia Madrid Spain Spain 0.7410277724266052
# Canberra Australia Moscow Russia Russia 0.7891087532043457
# Canberra Australia Oslo Norway Norway 0.670374870300293
# Canberra Australia Ottawa Canada Canada 0.7492120265960693
# Canberra Australia Paris France France 0.7159997820854187
# Canberra Australia Rome Italy Italy 0.6639474034309387
# Canberra Australia Stockholm Sweden Sweden 0.7069719433784485
# Canberra Australia Tehran Iran Iran 0.7551096677780151
# Canberra Australia Tokyo Japan Japan 0.7584798336029053
# Canberra Australia Athens Greece Greece 0.5700703263282776
# Canberra Australia Baghdad Iraq Mosul 0.6397026777267456
# Canberra Australia Bangkok Thailand Thailand 0.7358658909797668
# Canberra Australia Beijing China China 0.7823367118835449
# Canberra Australia Berlin Germany Germany 0.7441057562828064
# Canberra Australia Bern Switzerland Switzerland 0.5360245108604431
# Canberra Australia Cairo Egypt Egypt 0.6859405636787415
# Hanoi Vietnam Havana Cuba Cuba 0.68318110704422
# Hanoi Vietnam Helsinki Finland Finland 0.5732651948928833
# Hanoi Vietnam Islamabad Pakistan Pakistan 0.7073385715484619
# Hanoi Vietnam Kabul Afghanistan Afghanistan 0.7566174864768982
# Hanoi Vietnam London England Britain 0.5320746302604675
# Hanoi Vietnam Madrid Spain Spain 0.6318689584732056
# Hanoi Vietnam Moscow Russia Russia 0.6925882697105408
# Hanoi Vietnam Oslo Norway Norway 0.5300315618515015
# Hanoi Vietnam Ottawa Canada Canada 0.5802633762359619
# Hanoi Vietnam Paris France France 0.5821923613548279
# Hanoi Vietnam Rome Italy Italy 0.5208399295806885
# Hanoi Vietnam Stockholm Sweden Sweden 0.5909289717674255
# Hanoi Vietnam Tehran Iran Iran 0.7381365299224854
# Hanoi Vietnam Tokyo Japan Japan 0.6836792826652527
# Hanoi Vietnam Athens Greece Greece 0.5295498967170715
# Hanoi Vietnam Baghdad Iraq Iraq 0.7784610986709595
# Hanoi Vietnam Bangkok Thailand Thailand 0.6948618292808533
# Hanoi Vietnam Beijing China China 0.7234393954277039
# Hanoi Vietnam Berlin Germany Germany 0.5970560908317566
# Hanoi Vietnam Bern Switzerland Bern_NC 0.497576504945755
# Hanoi Vietnam Cairo Egypt Egypt 0.70243239402771
# Hanoi Vietnam Canberra Australia Australia 0.6169842481613159
# Havana Cuba Helsinki Finland Finland 0.6971813440322876
# Havana Cuba Islamabad Pakistan Pakistan 0.8123713731765747
# Havana Cuba Kabul Afghanistan Afghanistan 0.709291934967041
# Havana Cuba London England Britain 0.6217109560966492
# Havana Cuba Madrid Spain Spain 0.6929836273193359
# Havana Cuba Moscow Russia Russia 0.8237830400466919
# Havana Cuba Oslo Norway Norway 0.7030717134475708
# Havana Cuba Ottawa Canada Canada 0.6562575697898865
# Havana Cuba Paris France France 0.6843083500862122
# Havana Cuba Rome Italy Italy 0.6025850772857666
# Havana Cuba Stockholm Sweden Sweden 0.7303111553192139
# Havana Cuba Tehran Iran Iran 0.8635511994361877
# Havana Cuba Tokyo Japan Japan 0.7725999355316162
# Havana Cuba Athens Greece Greece 0.6323992013931274
# Havana Cuba Baghdad Iraq Iraq 0.7371043562889099
# Havana Cuba Bangkok Thailand Thailand 0.7950247526168823
# Havana Cuba Beijing China China 0.7704687118530273
# Havana Cuba Berlin Germany Germany 0.6952271461486816
# Havana Cuba Bern Switzerland Switzerland 0.5027072429656982
# Havana Cuba Cairo Egypt Egypt 0.7647275328636169
# Havana Cuba Canberra Australia Australia 0.6877732276916504
# Havana Cuba Hanoi Vietnam Viet_Nam 0.7109317183494568
# Helsinki Finland Islamabad Pakistan Pakistan 0.7291181087493896
# Helsinki Finland Kabul Afghanistan Afghan 0.6359645128250122
# Helsinki Finland London England Britain 0.6623522639274597
# Helsinki Finland Madrid Spain Spain 0.7539467811584473
# Helsinki Finland Moscow Russia Russia 0.7918932437896729
# Helsinki Finland Oslo Norway Norway 0.752316951751709
# Helsinki Finland Ottawa Canada Canada 0.6696932315826416
# Helsinki Finland Paris France France 0.7549223303794861
# Helsinki Finland Rome Italy Italy 0.6867628693580627
# Helsinki Finland Stockholm Sweden Sweden 0.8256369829177856
# Helsinki Finland Tehran Iran Iran 0.7732709646224976
# Helsinki Finland Tokyo Japan Japan 0.8030250668525696
# Helsinki Finland Athens Greece Greece 0.6566226482391357
# Helsinki Finland Baghdad Iraq Mosul 0.6506971716880798
# Helsinki Finland Bangkok Thailand Thailand 0.752036452293396
# Helsinki Finland Beijing China China 0.7568842172622681
# Helsinki Finland Berlin Germany Germany 0.7419475317001343
# Helsinki Finland Bern Switzerland Switzerland 0.5047451257705688
# Helsinki Finland Cairo Egypt Egypt 0.7302725911140442
# Helsinki Finland Canberra Australia Australia 0.6987783908843994
# Helsinki Finland Hanoi Vietnam Viet_Nam 0.6931596398353577
# Helsinki Finland Havana Cuba Cuba 0.6899740099906921
# Islamabad Pakistan Kabul Afghanistan Afghan 0.7332905530929565
# Islamabad Pakistan London England Britain 0.6056548357009888
# Islamabad Pakistan Madrid Spain Spain 0.6914530992507935
# Islamabad Pakistan Moscow Russia Russia 0.831927478313446
# Islamabad Pakistan Oslo Norway Norway 0.6795673966407776
# Islamabad Pakistan Ottawa Canada Canada 0.6421676874160767
# Islamabad Pakistan Paris France France 0.6944429874420166
# Islamabad Pakistan Rome Italy Italy 0.6518086791038513
# Islamabad Pakistan Stockholm Sweden Sweden 0.7098783850669861
# Islamabad Pakistan Tehran Iran Iran 0.8605974316596985
# Islamabad Pakistan Tokyo Japan Japan 0.7747603058815002
# Islamabad Pakistan Athens Greece Greece 0.6418431401252747
# Islamabad Pakistan Baghdad Iraq Iraqi 0.74811190366745
# Islamabad Pakistan Bangkok Thailand Thailand 0.7783140540122986
# Islamabad Pakistan Beijing China China 0.8079828023910522
# Islamabad Pakistan Berlin Germany Germany 0.7216539978981018
# Islamabad Pakistan Bern Switzerland Switzerland 0.49683260917663574
# Islamabad Pakistan Cairo Egypt Egypt 0.7879177927970886
# Islamabad Pakistan Canberra Australia Australia 0.7616326808929443
# Islamabad Pakistan Hanoi Vietnam Viet_Nam 0.7534357309341431
# Islamabad Pakistan Havana Cuba Cuba 0.7958502173423767
# Islamabad Pakistan Helsinki Finland Finland 0.693510115146637
# Kabul Afghanistan London England EURASIAN_NATURAL_RESOURCES_CORP. 0.5547584891319275
# Kabul Afghanistan Madrid Spain Spain 0.6156005859375
# Kabul Afghanistan Moscow Russia Russia 0.7199913859367371
# Kabul Afghanistan Oslo Norway Norway 0.5757439136505127
# Kabul Afghanistan Ottawa Canada Ontario 0.551449179649353
# Kabul Afghanistan Paris France France 0.6088169813156128
# Kabul Afghanistan Rome Italy Italy 0.5375437140464783
# Kabul Afghanistan Stockholm Sweden Sweden 0.5849066376686096
# Kabul Afghanistan Tehran Iran Iran 0.7807148694992065
# Kabul Afghanistan Tokyo Japan Japan 0.6695030927658081
# Kabul Afghanistan Athens Greece Greece 0.5494598746299744
# Kabul Afghanistan Baghdad Iraq Iraq 0.8607981204986572
# Kabul Afghanistan Bangkok Thailand Thailand 0.6891649961471558
# Kabul Afghanistan Beijing China China 0.6837893128395081
# Kabul Afghanistan Berlin Germany Germany 0.6105466485023499
# Kabul Afghanistan Bern Switzerland Bern_NC 0.42338475584983826
# Kabul Afghanistan Cairo Egypt Egypt 0.6705427169799805
# Kabul Afghanistan Canberra Australia Mr_Rudd 0.6034548282623291
# Kabul Afghanistan Hanoi Vietnam Vietnam 0.7514270544052124
# Kabul Afghanistan Havana Cuba Cuba 0.7205934524536133
# Kabul Afghanistan Helsinki Finland Finland 0.5768634080886841
# Kabul Afghanistan Islamabad Pakistan Pakistan 0.7463313341140747
# London England Madrid Spain Spain 0.6087852716445923
# London England Moscow Russia Russia 0.6502716541290283
# London England Oslo Norway Norway 0.5689430236816406
# London England Ottawa Canada Canada 0.5256192684173584
# London England Paris France France 0.6676379442214966
# London England Rome Italy Italy 0.5715967416763306
# London England Stockholm Sweden Sweden 0.6610991358757019
# London England Tehran Iran Iran 0.6443779468536377
# London England Tokyo Japan Japan 0.6889491081237793
# London England Athens Greece Greece 0.5293721556663513
# London England Baghdad Iraq Iraqi 0.5854012966156006
# London England Bangkok Thailand Thailand 0.6264095306396484
# London England Beijing China China 0.6181186437606812
# London England Berlin Germany Germany 0.5952338576316833
# London England Bern Switzerland Hanover 0.4530367851257324
# London England Cairo Egypt Egypt 0.5641492009162903
# London England Canberra Australia Australia 0.6054890751838684
# London England Hanoi Vietnam Viet_Nam 0.5810423493385315
# London England Havana Cuba Cuba 0.5627489686012268
# London England Helsinki Finland Finland 0.6215444207191467
# London England Islamabad Pakistan Pakistan 0.6654638051986694
# London England Kabul Afghanistan Afghan 0.5733120441436768
# Madrid Spain Moscow Russia Russia 0.8136782050132751
# Madrid Spain Oslo Norway Norway 0.7230186462402344
# Madrid Spain Ottawa Canada Canada 0.6814807653427124
# Madrid Spain Paris France France 0.7609946727752686
# Madrid Spain Rome Italy Italy 0.7004685997962952
# Madrid Spain Stockholm Sweden Sweden 0.7822341918945312
# Madrid Spain Tehran Iran Iran 0.7588755488395691
# Madrid Spain Tokyo Japan Japan 0.8010953664779663
# Madrid Spain Athens Greece Greece 0.6898481845855713
# Madrid Spain Baghdad Iraq Mosul 0.6579501628875732
# Madrid Spain Bangkok Thailand Thailand 0.7701379656791687
# Madrid Spain Beijing China China 0.7744125723838806
# Madrid Spain Berlin Germany Germany 0.7501679062843323
# Madrid Spain Bern Switzerland Switzerland 0.48540329933166504
# Madrid Spain Cairo Egypt Egypt 0.7326661348342896
# Madrid Spain Canberra Australia Australia 0.706891655921936
# Madrid Spain Hanoi Vietnam Viet_Nam 0.7359682321548462
# Madrid Spain Havana Cuba Cuba 0.7222690582275391
# Madrid Spain Helsinki Finland Finland 0.7638067007064819
# Madrid Spain Islamabad Pakistan Pakistan 0.7176594734191895
# Madrid Spain Kabul Afghanistan Herat 0.6163390874862671
# Madrid Spain London England Britain 0.6784723401069641
# Moscow Russia Oslo Norway Norway 0.7606369853019714
# Moscow Russia Ottawa Canada Canada 0.7216338515281677
# Moscow Russia Paris France France 0.8050409555435181
# Moscow Russia Rome Italy Italy 0.7074910998344421
# Moscow Russia Stockholm Sweden Sweden 0.805422842502594
# Moscow Russia Tehran Iran Iran 0.8776857256889343
# Moscow Russia Tokyo Japan Japan 0.8773881196975708
# Moscow Russia Athens Greece Greece 0.6744522452354431
# Moscow Russia Baghdad Iraq Mosul 0.7155514359474182
# Moscow Russia Bangkok Thailand Thailand 0.842517077922821
# Moscow Russia Beijing China China 0.8687477707862854
# Moscow Russia Berlin Germany Germany 0.777425229549408
# Moscow Russia Bern Switzerland Switzerland 0.48516905307769775
# Moscow Russia Cairo Egypt Egypt 0.8018569350242615
# Moscow Russia Canberra Australia Australia 0.7593483328819275
# Moscow Russia Hanoi Vietnam Viet_Nam 0.7691516280174255
# Moscow Russia Havana Cuba Cuba 0.7886744737625122
# Moscow Russia Helsinki Finland Finland 0.7779611945152283
# Moscow Russia Islamabad Pakistan Pakistan 0.8369842171669006
# Moscow Russia Kabul Afghanistan Afghan 0.7208551168441772
# Moscow Russia London England Britain 0.7028805613517761
# Moscow Russia Madrid Spain Spain 0.7905085682868958
# Oslo Norway Ottawa Canada Canada 0.6809220910072327
# Oslo Norway Paris France France 0.731560468673706
# Oslo Norway Rome Italy Italy 0.6817960143089294
# Oslo Norway Stockholm Sweden Sweden 0.8222189545631409
# Oslo Norway Tehran Iran Iran 0.7522603273391724
# Oslo Norway Tokyo Japan Japan 0.7785905003547668
# Oslo Norway Athens Greece Greece 0.631377100944519
# Oslo Norway Baghdad Iraq Mosul 0.6509431600570679
# Oslo Norway Bangkok Thailand Thailand 0.7387948632240295
# Oslo Norway Beijing China China 0.7163815498352051
# Oslo Norway Berlin Germany Germany 0.721919059753418
# Oslo Norway Bern Switzerland Switzerland 0.5127914547920227
# Oslo Norway Cairo Egypt Egypt 0.6662337183952332
# Oslo Norway Canberra Australia Australia 0.6750895977020264
# Oslo Norway Hanoi Vietnam Viet_Nam 0.6644772887229919
# Oslo Norway Havana Cuba Cuba 0.687487006187439
# Oslo Norway Helsinki Finland Finland 0.7806000113487244
# Oslo Norway Islamabad Pakistan Pakistan 0.7272922992706299
# Oslo Norway Kabul Afghanistan Afghan 0.6326096057891846
# Oslo Norway London England Britain 0.6456331014633179
# Oslo Norway Madrid Spain Spain 0.7302019596099854
# Oslo Norway Moscow Russia Russia 0.7961567044258118
# Ottawa Canada Paris France France 0.6939915418624878
# Ottawa Canada Rome Italy Italy 0.6293004751205444
# Ottawa Canada Stockholm Sweden Sweden 0.6766182780265808
# Ottawa Canada Tehran Iran Iran 0.7350882291793823
# Ottawa Canada Tokyo Japan Japan 0.7678512930870056
# Ottawa Canada Athens Greece Greece 0.6065005660057068
# Ottawa Canada Baghdad Iraq Mosul 0.6319442391395569
# Ottawa Canada Bangkok Thailand Thailand 0.711449384689331
# Ottawa Canada Beijing China China 0.7349016070365906
# Ottawa Canada Berlin Germany Germany 0.7051476836204529
# Ottawa Canada Bern Switzerland Switzerland 0.5049155950546265
# Ottawa Canada Cairo Egypt Egypt 0.6439140439033508
# Ottawa Canada Canberra Australia Australia 0.7822460532188416
# Ottawa Canada Hanoi Vietnam Viet_Nam 0.6644909381866455
# Ottawa Canada Havana Cuba Cuba 0.6701314449310303
# Ottawa Canada Helsinki Finland Finland 0.6524146795272827
# Ottawa Canada Islamabad Pakistan Pakistan 0.6897966861724854
# Ottawa Canada Kabul Afghanistan Afghan 0.5671024918556213
# Ottawa Canada London England UK 0.636229395866394
# Ottawa Canada Madrid Spain Spain 0.7073505520820618
# Ottawa Canada Moscow Russia Russia 0.7822917699813843
# Ottawa Canada Oslo Norway Norway 0.6731112599372864
# Paris France Rome Italy Italy 0.7143560647964478
# Paris France Stockholm Sweden Sweden 0.7924454808235168
# Paris France Tehran Iran Iran 0.7771149277687073
# Paris France Tokyo Japan Japan 0.8167769312858582
# Paris France Athens Greece Greece 0.637862503528595
# Paris France Baghdad Iraq Mosul 0.6609416007995605
# Paris France Bangkok Thailand Thailand 0.750830888748169
# Paris France Beijing China China 0.7644792199134827
# Paris France Berlin Germany Germany 0.7901254892349243
# Paris France Bern Switzerland Switzerland 0.48789724707603455
# Paris France Cairo Egypt Egypt 0.7062106728553772
# Paris France Canberra Australia Australia 0.7059868574142456
# Paris France Hanoi Vietnam Viet_Nam 0.7121440172195435
# Paris France Havana Cuba Cuba 0.7165152430534363
# Paris France Helsinki Finland Finland 0.7654454112052917
# Paris France Islamabad Pakistan Pakistan 0.7565772533416748
# Paris France Kabul Afghanistan Afghan 0.669233500957489
# Paris France London England Britain 0.7368935942649841
# Paris France Madrid Spain Spain 0.7776166200637817
# Paris France Moscow Russia Russia 0.834155797958374
# Paris France Oslo Norway Norway 0.7406579852104187
# Paris France Ottawa Canada Canada 0.6765265464782715
# Rome Italy Stockholm Sweden Sweden 0.7650186419487
# Rome Italy Tehran Iran Iran 0.722395658493042
# Rome Italy Tokyo Japan Japan 0.7659536600112915
# Rome Italy Athens Greece Greece 0.6582207083702087
# Rome Italy Baghdad Iraq Iraqi 0.6279985904693604
# Rome Italy Bangkok Thailand Thailand 0.7367711067199707
# Rome Italy Beijing China China 0.738448977470398
# Rome Italy Berlin Germany Germany 0.7924448251724243
# Rome Italy Bern Switzerland Switzerland 0.5860610604286194
# Rome Italy Cairo Egypt Egypt 0.6917356848716736
# Rome Italy Canberra Australia Australia 0.7035337090492249
# Rome Italy Hanoi Vietnam Viet_Nam 0.6718668937683105
# Rome Italy Havana Cuba Cuba 0.6652557253837585
# Rome Italy Helsinki Finland Finland 0.7448439598083496
# Rome Italy Islamabad Pakistan Pakistan 0.7375875115394592
# Rome Italy Kabul Afghanistan Afghan 0.6492578387260437
# Rome Italy London England UK 0.623056948184967
# Rome Italy Madrid Spain Spain 0.7485761046409607
# Rome Italy Moscow Russia Russia 0.7561044692993164
# Rome Italy Oslo Norway Norway 0.714056670665741
# Rome Italy Ottawa Canada Canada 0.6508231163024902
# Rome Italy Paris France France 0.7486129403114319
# Stockholm Sweden Tehran Iran Iran 0.7674086689949036
# Stockholm Sweden Tokyo Japan Japan 0.8198506832122803
# Stockholm Sweden Athens Greece Greece 0.6445161700248718
# Stockholm Sweden Baghdad Iraq Mosul 0.674405574798584
# Stockholm Sweden Bangkok Thailand Thailand 0.7606315612792969
# Stockholm Sweden Beijing China China 0.7469374537467957
# Stockholm Sweden Berlin Germany Germany 0.7917513251304626
# Stockholm Sweden Bern Switzerland Switzerland 0.5347192287445068
# Stockholm Sweden Cairo Egypt Egypt 0.7208700776100159
# Stockholm Sweden Canberra Australia Australia 0.6957679986953735
# Stockholm Sweden Hanoi Vietnam Viet_Nam 0.6892715096473694
# Stockholm Sweden Havana Cuba Cuba 0.7302880883216858
# Stockholm Sweden Helsinki Finland Finland 0.8304908275604248
# Stockholm Sweden Islamabad Pakistan Pakistan 0.7434418797492981
# Stockholm Sweden Kabul Afghanistan Afghan 0.6353626847267151
# Stockholm Sweden London England Britain 0.6828158497810364
# Stockholm Sweden Madrid Spain Spain 0.779670238494873
# Stockholm Sweden Moscow Russia Russia 0.8211641311645508
# Stockholm Sweden Oslo Norway Norway 0.7942283153533936
# Stockholm Sweden Ottawa Canada Canada 0.6786275506019592
# Stockholm Sweden Paris France France 0.7826050519943237
# Stockholm Sweden Rome Italy Italy 0.7247942686080933
# Tehran Iran Tokyo Japan Japan 0.7860096096992493
# Tehran Iran Athens Greece Greece 0.657496988773346
# Tehran Iran Baghdad Iraq Iraq 0.7758105993270874
# Tehran Iran Bangkok Thailand Thailand 0.8036007881164551
# Tehran Iran Beijing China China 0.8198245763778687
# Tehran Iran Berlin Germany Germany 0.6948079466819763
# Tehran Iran Bern Switzerland coach_Bobby_Curlings 0.5130249261856079
# Tehran Iran Cairo Egypt Egypt 0.8029150366783142
# Tehran Iran Canberra Australia Australia 0.7018476724624634
# Tehran Iran Hanoi Vietnam Viet_Nam 0.7534172534942627
# Tehran Iran Havana Cuba Cuba 0.8335038423538208
# Tehran Iran Helsinki Finland Finland 0.6951573491096497
# Tehran Iran Islamabad Pakistan Pakistan 0.8478929996490479
# Tehran Iran Kabul Afghanistan Afghan 0.7277184128761292
# Tehran Iran London England Britain 0.6258983612060547
# Tehran Iran Madrid Spain Spain 0.7075908184051514
# Tehran Iran Moscow Russia Russia 0.8626428842544556
# Tehran Iran Oslo Norway Norway 0.7017198204994202
# Tehran Iran Ottawa Canada Canada 0.6508047580718994
# Tehran Iran Paris France France 0.680817186832428
# Tehran Iran Rome Italy Italy 0.6060930490493774
# Tehran Iran Stockholm Sweden Sweden 0.7119143009185791
# Tokyo Japan Athens Greece Greece 0.6701824069023132
# Tokyo Japan Baghdad Iraq Iraq 0.7005369663238525
# Tokyo Japan Bangkok Thailand Thailand 0.8329715132713318
# Tokyo Japan Beijing China China 0.8427804708480835
# Tokyo Japan Berlin Germany Germany 0.7607141733169556
# Tokyo Japan Bern Switzerland Switzerland 0.48349660634994507
# Tokyo Japan Cairo Egypt Egypt 0.760470986366272
# Tokyo Japan Canberra Australia Australia 0.7408983111381531
# Tokyo Japan Hanoi Vietnam Viet_Nam 0.7667806148529053
# Tokyo Japan Havana Cuba Cuba 0.766819417476654
# Tokyo Japan Helsinki Finland Finland 0.7938802242279053
# Tokyo Japan Islamabad Pakistan Pakistan 0.794365644454956
# Tokyo Japan Kabul Afghanistan Afghanistan 0.6684214472770691
# Tokyo Japan London England Britain 0.7236440181732178
# Tokyo Japan Madrid Spain Spain 0.7904621362686157
# Tokyo Japan Moscow Russia Russia 0.8863135576248169
# Tokyo Japan Oslo Norway Norway 0.7558910846710205
# Tokyo Japan Ottawa Canada Canada 0.7142959237098694
# Tokyo Japan Paris France France 0.7889979481697083
# Tokyo Japan Rome Italy Italy 0.6944944858551025
# Tokyo Japan Stockholm Sweden Sweden 0.8109342455863953
# Tokyo Japan Tehran Iran Iran 0.8259532451629639
