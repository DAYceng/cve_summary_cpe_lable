import os, sys, inspect
import re
import nltk
import numpy as np
import itertools

from nltk.tokenize import MWETokenizer
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import conlltags2tree, tree2conlltags

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def my_str_lower(data):
    data = data.lower()
    return data

def mergelist(l1,l2):
    '''
    将两个长度相同的标签列表合并生成完整的vp标签列表
    :param l1:
    :param l2:
    :return:
    '''
    l3 = []
    for i in range(len(l1)):
        if l1[i] != 'O':
            l3.append(l1[i])
        elif l2[i] != 'O':
            l3.append(l2[i])
        else:
            l3.append('O')
    # print(l3)
    return l3

def summary_deal(summary_raw,setcase):
    '''
    用于处理原始summary的函数
    :param summary_raw: 从summary列表遍历得到一条summary并传入
    :param setcase:设置输出summary的大小写形式，0为全小写，1为保留原始大小写
    :return: 返回一个分隔好的字符串形式的summary
    '''
    summary_backup = summary_raw  #先将原始summary（即summary_i ）暂存

    sentence = summary_raw.replace('_',' ')
    summary_data = re.findall(r"[\w']+|[\"',!?;*_]", sentence)#将小写后的summary标点分割
    # print(summary_data)
    summary_data = ' '.join(summary_data)

    # print(summary_data)

    summary_lower = summary_data.lower()  #变为小写
    # print(summary_lower)
    summary_yuan = ' '.join(summary_data)
    # print(summary_yuan)
    if setcase == 0:
        return summary_lower
    elif setcase == 1:
        return summary_data

def postags(summary_token,output):
    '''
    使用nltk.pos_tag对分词之后的summary进行词性标注
    :param output:
    选择输出形式，
    输出原始格式数据（0）：[('buffer', 'NN'), ('overflow', 'NN'), ...,(...)]
    输出词性标签（1）：['NN', 'NN',...,'?']
    输出chunking标签（2）
    :param data:经过'分词-去除特殊符号-小写'处理的summary数据
    :return:列表返回值
    '''
    ne_chunked_tags = []
    pos_tags = []
    pos_raw = []

    tokenized_sentences = summary_token.split(" ") #nltk.word_tokenize(data)
    # print(tokenized_sentences)
    pos_res = nltk.pos_tag(tokenized_sentences)
    pos_res_tup = pos_res[0]   # 得到元组

    tree = ne_chunk(pos_res)   # 使用nltk的chunk工具获得chunk的树结构
    # print(tree)
    iob_tags_listtup = tree2conlltags(tree) # 解析树，获得chunktags的元组列表
    # print(iob_tags_listtup)
    # print(iob_tags_listtup)
    for i in range(len(iob_tags_listtup)):
        # print(iob_tags_listtup[i])
        tmp_tup = iob_tags_listtup[i]
        # print(tmp_tup[2])
        pos_raw.append(tmp_tup)
        pos_tags.append(tmp_tup[1])
        ne_chunked_tags.append(tmp_tup[2])
    # print(ne_chunked_tags)

    if output == 0:
        return pos_raw
    elif output == 1:
        return pos_tags
    elif output == 2:
        return ne_chunked_tags


def summary_lable_process(data, lable,model):
    '''
    用于为summary打标签的函数
    :param data: 经过'分词-去除特殊符号-小写'处理的summary数据
    :param label: 这里的label指CPEs,即vendor或product（序列）
    :param model:选择标签类型，0时为vendor打标签，1时为product打标签
    :return:
    list_x--经过处理并保留大小写特征和标点符号的summary列表
    list_y--标签列表
    '''
    if model == 0:
        ioblabel_1 = 'B-VENDOR'
        ioblabel_2 = 'I-VENDOR'
    elif model == 1:
        ioblabel_1 = 'B-PRODUCT'
        ioblabel_2 = 'I-PRODUCT'
    lable_O = 'O'


    list_data = data.lower().split(" ")
    list_lable = lable.lower().split(" ")

    list_x = data.split(" ")
    list_y = []

    i = 0             #控制遍历summary的次数
    count = 0         #偏移量，在遍历列表时移动指针（列表元素位置）
    while(True):
        if i >= len(list_data):
            break
#j在匹配条件满足时才会增加，否则会每次被重置为0，因此其可作为遍历多词vp的依据
        for j in range(len(list_lable)):
            if i + j >= len(list_data):
                break #避免列表数据溢出
                      #遍历获取summary的词与CPE进行比较（此时处理的是vendor还是product取决于输入数据）
                      #取CPEs即vendor/product中的第一个词进行比较（j == 0）
            if list_data[i] == list_lable[j] and j == 0:
                count = count + 1
                # print("find B: {}".format(list_lable[j]))
                list_y.append(ioblabel_1)
#j若不为0，则表示来到vp中的第二个词
            elif list_data[i+j] == list_lable[j] and j != 0:
                count = count + 1
                # print("find I: {}".format(list_lable[j]))
                list_y.append(ioblabel_2)

            else:
                count = count + 1
                list_y.append(lable_O) #若未匹配上，先打o标签，并且记录偏移量

                tmp_count = j
                while tmp_count != 0:
                    list_y[i+tmp_count-1] = lable_O #将之前未完全匹配的B或I标签更换为o
                    tmp_count = tmp_count - 1

                break

        # 循环更新，清空偏移量
        i = i + count
        count = 0

    # print(list_data)
    # print(list_y)
    # print(list_lable)

    return list_x, list_y


if __name__ == '__main__':
    # 测试用数据
    # summary_token = "aa bb xxx CC dd ee ff ee cc dd ee ff"
    # summary_token2 = 'The /etc/profile.d/60alias.sh script in the Mandriva bash package for Bash 2.05b, 3.0, 3.2, 3.2.48, and 4.0 enables the --show-control-chars option in LS_OPTIONS, which allows local users to send escape sequences to terminal emulators, or hide the existence of a file, via a crafted filename.'
    # vendor_lable = "xxx"
    # product_lable = "cc DD ee ff"

    # summary_raw = 'The /etc/profile.d/60alias.sh script in the Mandriva bash package for Bash 2.05b, 3.0, 3.2, 3.2.48, and 4.0 enables the --show-control-chars option in LS_OPTIONS, which allows local users to send escape sequences to terminal emulators, or hide the existence of a file, via a crafted filename.'
    # summary_token = summary_deal(summary_raw,1)
    # vendor_lable = 'gnu'
    # product_lable = 'Bash'

    summary = ['Buffer overflow in the Windows Redirector function in Microsoft Windows XP allows local users to execute arbitrary code via a long parameter.', 'Microsoft PowerPoint 2000 in Office 2000 SP3 has an interaction with Internet Explorer that allows remote attackers to obtain sensitive information via a PowerPoint presentation that attempts to access objects in the Temporary Internet Files Folder (TIFF).', 'The kernel in Microsoft Windows XP SP2, Windows Server 2003 SP2, Windows Vista SP2, Windows Server 2008 SP2, R2, and R2 SP1, and Windows 7 Gold and SP1 does not properly load structured exception handling tables, which allows context-dependent attackers to bypass the SafeSEH security feature by leveraging a Visual C++ .NET 2003 application, aka "Windows Kernel SafeSEH Bypass Vulnerability."', 'Cross-site scripting (XSS) vulnerability in Help and Support Center for Microsoft Windows Me allows remote attackers to execute arbitrary script in the Local Computer security context via an hcp:// URL with the malicious script in the topic parameter.', 'The /etc/profile.d/60alias.sh script in the Mandriva bash package for Bash 2.05b, 3.0, 3.2, 3.2.48, and 4.0 enables the --show-control-chars option in LS_OPTIONS, which allows local users to send escape sequences to terminal emulators, or hide the existence of a file, via a crafted filename.', 'Integer overflow in JsArrayFunctionHeapSort function used by Windows Script Engine for JScript (JScript.dll) on various Windows operating system allows remote attackers to execute arbitrary code via a malicious web page or HTML e-mail that uses a large array index value that enables a heap-based buffer overflow attack.', 'The Windows Forms (aka WinForms) component in Microsoft .NET Framework 1.0 SP3, 1.1 SP1, 2.0 SP2, 3.0 SP2, 4, and 4.5 does not properly initialize memory arrays, which allows remote attackers to obtain sensitive information via (1) a crafted XAML browser application (XBAP) or (2) a crafted .NET Framework application that leverages a pointer to an unmanaged memory location, aka "System Drawing Information Disclosure Vulnerability." Per http://technet.microsoft.com/en-us/security/bulletin/ms13-004 Microsoft .NET Framework 3.0 Service Pack 2 is not vulnerable.', 'libuser before 0.57 uses a cleartext password value of (1) !! or (2) x for new LDAP user accounts, which makes it easier for remote attackers to obtain access by specifying one of these values.', 'The Windows Error Reporting (WER) component in Microsoft Windows 8, Windows 8.1, Windows Server 2012 Gold and R2, and Windows RT Gold and 8.1 allows local users to bypass the Protected Process Light protection mechanism and read the contents of arbitrary process-memory locations by leveraging administrative privileges, aka "Windows Error Reporting Security Feature Bypass Vulnerability."', 'The AhcVerifyAdminContext function in ahcache.sys in the Application Compatibility component in Microsoft Windows 7 SP1, Windows Server 2008 R2 SP1, Windows 8, Windows 8.1, Windows Server 2012 Gold and R2, and Windows RT Gold and 8.1 does not verify that an impersonation token is associated with an administrative account, which allows local users to gain privileges by running AppCompatCache.exe with a crafted DLL file, aka MSRC ID 20544 or "Microsoft Application Compatibility Infrastructure Elevation of Privilege Vulnerability."', 'Unspecified vulnerability in winmm.dll in Windows Multimedia Library in Windows Media Player (WMP) in Microsoft Windows XP SP2 and SP3, Server 2003 SP2, Vista SP2, and Server 2008 SP2 allows remote attackers to execute arbitrary code via a crafted MIDI file, aka "MIDI Remote Code Execution Vulnerability."', 'The default .htaccess scripts for Bugzilla 2.14.x before 2.14.5, 2.16.x before 2.16.2, and 2.17.x before 2.17.3 do not include filenames for backup copies of the localconfig file that are made from editors such as vi and Emacs, which could allow remote attackers to obtain a database password by directly accessing the backup file.', 'Unspecified vulnerability in DirectShow in DirectX in Microsoft Windows XP SP2 and SP3, Windows Server 2003 SP2, Windows Vista SP2, Windows Server 2008 SP2, R2, and R2 SP1, and Windows 7 Gold and SP1 allows remote attackers to execute arbitrary code via a crafted media file, related to Quartz.dll, Qdvd.dll, closed captioning, and the Line21 DirectShow filter, aka "DirectShow Remote Code Execution Vulnerability."', 'gsinterf.c in bmv 1.2 and earlier allows local users to overwrite arbitrary files via a symlink attack on temporary files. For the stable distribution this problem has been fixed in version 1.2-14.2. For the unstable distribution this problem has been fixed in version 1.2-17.', 'The Client/Server Run-time Subsystem (aka CSRSS) in the Win32 subsystem in Microsoft Windows XP SP2 and SP3, Server 2003 SP2, Vista SP2, and Server 2008 SP2, when a Chinese, Japanese, or Korean system locale is used, can access uninitialized memory during the processing of Unicode characters, which allows local users to gain privileges via a crafted application, aka "CSRSS Elevation of Privilege Vulnerability."', 'Multiple buffer overflows in Gaim 0.75 allow remote attackers to cause a denial of service and possibly execute arbitrary code via (1) octal encoding in yahoo_decode that causes a null byte to be written beyond the buffer, (2) octal encoding in yahoo_decode that causes a pointer to reference memory beyond the terminating null byte, (3) a quoted printable string to the gaim_quotedp_decode MIME decoder that causes a null byte to be written beyond the buffer, and (4) quoted printable encoding in gaim_quotedp_decode that causes a pointer to reference memory beyond the terminating null byte.', 'The ShellAbout API call in Korean Input Method Editor (IME) in Korean versions of Microsoft Windows XP SP1 and SP2, Windows Server 2003 up to SP1, and Office 2003, allows local users to gain privileges by launching the "shell about dialog box" and clicking the "End-User License Agreement" link, which executes Notepad with the privileges of the program that displays the about box.', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2015. Notes: none.', 'The Group Policy Security Configuration policy implementation in Microsoft Windows Server 2003 SP2, Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8, Windows 8.1, Windows Server 2012 Gold and R2, and Windows RT Gold and 8.1 allows man-in-the-middle attackers to disable a signing requirement and trigger a revert-to-default action by spoofing domain-controller responses, aka "Group Policy Security Feature Bypass Vulnerability."', 'The CryptProtectMemory function in cng.sys (aka the Cryptography Next Generation driver) in the kernel-mode drivers in Microsoft Windows Server 2003 SP2, Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8, Windows 8.1, Windows Server 2012 Gold and R2, and Windows RT Gold and 8.1, when the CRYPTPROTECTMEMORY_SAME_LOGON option is used, does not check an impersonation token\'s level, which allows local users to bypass intended decryption restrictions by leveraging a service that (1) has a named-pipe planting vulnerability or (2) uses world-readable shared memory for encrypted data, aka "CNG Security Feature Bypass Vulnerability" or MSRC ID 20707.', 'The XSLT component in Apache Camel before 2.11.4 and 2.12.x before 2.12.3 allows remote attackers to read arbitrary files and possibly have other unspecified impact via an XML document containing an external entity declaration in conjunction with an entity reference, related to an XML External Entity (XXE) issue.', 'mrxdav.sys (aka the WebDAV driver) in the kernel-mode drivers in Microsoft Windows Server 2003 SP2, Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8, Windows 8.1, Windows Server 2012 Gold and R2, and Windows RT Gold and 8.1 allows local users to bypass an impersonation protection mechanism, and obtain privileges for redirection of WebDAV requests, via a crafted application, aka "WebDAV Elevation of Privilege Vulnerability."', 'Apache Tomcat 6.0.0 through 6.0.15 processes parameters in the context of the wrong request when an exception occurs during parameter processing, which might allow remote attackers to obtain sensitive information, as demonstrated by disconnecting during this processing in order to trigger the exception.', 'The Microsoft Anti-Cross Site Scripting (AntiXSS) Library 3.x and 4.0 does not properly evaluate characters after the detection of a Cascading Style Sheets (CSS) escaped character, which allows remote attackers to conduct cross-site scripting (XSS) attacks via HTML input, aka "AntiXSS Library Bypass Vulnerability."', 'Microsoft .NET Framework 1.0 SP3, 1.1 SP1, 2.0 SP2, 3.0 SP2, 3.5, 3.5.1, 4, and 4.5 does not properly validate the permissions of objects in memory, which allows remote attackers to execute arbitrary code via (1) a crafted XAML browser application (XBAP) or (2) a crafted .NET Framework application, aka "Double Construction Vulnerability."', 'Buffer overflow in the Telnet service in Microsoft Windows Server 2003 SP2, Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8, Windows 8.1, and Windows Server 2012 Gold and R2 allows remote attackers to execute arbitrary code via crafted packets, aka "Windows Telnet Service Buffer Overflow Vulnerability."', 'Microsoft Windows Server 2003 SP2, Server 2008 SP2 and R2 SP1, and Server 2012 Gold and R2 allow remote attackers to cause a denial of service (system hang and RADIUS outage) via crafted username strings to (1) Internet Authentication Service (IAS) or (2) Network Policy Server (NPS), aka "Network Policy Server RADIUS Implementation Denial of Service Vulnerability."', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER.  ConsultIDs: none.  Reason: This candidate was withdrawn by its CNA.  Further investigation showed that it was not a security issue.  Notes: none.', 'Untrusted search path vulnerability in the Windows Object Packager configuration in Microsoft Windows XP SP2 and SP3 and Server 2003 SP2 allows local users to gain privileges via a Trojan horse executable file in the current working directory, as demonstrated by a directory that contains a file with an embedded packaged object, aka "Object Packager Insecure Executable Launching Vulnerability." Per: http://technet.microsoft.com/en-us/security/bulletin/ms12-002\r\n\r\n\'The vulnerability could allow remote code execution if a user opens a legitimate file with an embedded packaged object that is located in the same network directory as a specially crafted executable file.\' Per: http://cwe.mitre.org/data/definitions/426.html\r\n\r\n\'CWE-426: Untrusted Search Path\'', 'Microsoft Edge allows remote attackers to execute arbitrary code via unspecified vectors, aka "Microsoft Edge Memory Corruption Vulnerability."', 'Buffer overflow in the Web Client service (WebClnt.dll) for Microsoft Windows XP SP1 and SP2, and Server 2003 up to SP1, allows remote authenticated users or Guests to execute arbitrary code via crafted RPC requests, a different vulnerability than CVE-2005-1207.', 'The XSLT component in Apache Camel 2.11.x before 2.11.4, 2.12.x before 2.12.3, and possibly earlier versions allows remote attackers to execute arbitrary Java methods via a crafted message.', 'Linux kernel 2.4.10 through 2.4.21-pre4 does not properly handle the O_DIRECT feature, which allows local attackers with write privileges to read portions of previously deleted files, or cause file system corruption.', 'Cross-site scripting (XSS) vulnerability in the com_search module for Joomla! 1.0.x through 1.0.15 allows remote attackers to inject arbitrary web script or HTML via the ordering parameter to index.php.', 'The sandbox implementation in Microsoft Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8, Windows 8.1, Windows Server 2012 Gold and R2, Windows RT Gold and 8.1, and Windows 10 Gold and 1511 mishandles reparse points, which allows local users to gain privileges via a crafted application, aka "Windows Mount Point Elevation of Privilege Vulnerability," a different vulnerability than CVE-2016-0007.', 'ViewVC before 1.1.3 composes the root listing view without using the authorizer for each root, which might allow remote attackers to discover private root names by reading this view.', 'The sandbox implementation in Microsoft Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8, Windows 8.1, Windows Server 2012 Gold and R2, Windows RT Gold and 8.1, and Windows 10 Gold and 1511 mishandles reparse points, which allows local users to gain privileges via a crafted application, aka "Windows Mount Point Elevation of Privilege Vulnerability," a different vulnerability than CVE-2016-0006.', 'The graphics device interface in Microsoft Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8, Windows 8.1, Windows Server 2012 Gold and R2, and Windows RT Gold and 8.1 allows remote attackers to bypass the ASLR protection mechanism via unspecified vectors, aka "Windows GDI32.dll ASLR Bypass Vulnerability."', 'Heap-based buffer overflow in the encodeURI and decodeURI functions in the kjs JavaScript interpreter engine in KDE 3.2.0 through 3.5.0 allows remote attackers to execute arbitrary code via a crafted, UTF-8 encoded URI.', 'Microsoft Internet Explorer 9 does not properly handle the creation and initialization of string objects, which allows remote attackers to read data from arbitrary process-memory locations via a crafted web site, aka "Null Byte Information Disclosure Vulnerability."', 'Incomplete blacklist vulnerability in the Windows Packager configuration in Microsoft Windows XP SP2 and SP3, Windows Server 2003 SP2, Windows Vista SP2, Windows Server 2008 SP2, R2, and R2 SP1, and Windows 7 Gold and SP1 allows remote attackers to execute arbitrary code via a crafted ClickOnce application in a Microsoft Office document, related to .application files, aka "Assembly Execution Vulnerability."', 'Microsoft SharePoint Server 2013 SP1 and SharePoint Foundation 2013 SP1 allow remote authenticated users to bypass intended Access Control Policy restrictions and conduct cross-site scripting (XSS) attacks by modifying a webpart, aka "Microsoft SharePoint Security Feature Bypass," a different vulnerability than CVE-2015-6117.', 'Cross-site scripting (XSS) vulnerability in Microsoft System Center Operations Manager 2007 SP1 and R2 allows remote attackers to inject arbitrary web script or HTML via crafted input, aka "System Center Operations Manager Web Console XSS Vulnerability," a different vulnerability than CVE-2013-0009.', 'Microsoft .NET Framework 2.0 SP2 and 3.5.1 does not properly calculate the length of an unspecified buffer, which allows remote attackers to execute arbitrary code via (1) a crafted XAML browser application (aka XBAP), (2) a crafted ASP.NET application, or (3) a crafted .NET Framework application, aka ".NET Framework Heap Corruption Vulnerability."', "The ima_lsm_rule_init function in security/integrity/ima/ima_policy.c in the Linux kernel before 2.6.37, when the Linux Security Modules (LSM) framework is disabled, allows local users to bypass Integrity Measurement Architecture (IMA) rules in opportunistic circumstances by leveraging an administrator's addition of an IMA rule for LSM.", 'Unspecified vulnerability in Microsoft Exchange allows remote attackers to execute arbitrary code via e-mail messages with crafted (1) vCal or (2) iCal Calendar properties.', 'Cross-site scripting (XSS) vulnerability in Outlook Web Access (OWA) in Microsoft Exchange Server 2013 PS1, 2013 Cumulative Update 10, and 2016 allows remote attackers to inject arbitrary web script or HTML via a crafted URL, aka "Exchange Spoofing Vulnerability."', 'Cross-site scripting (XSS) vulnerability in Outlook Web Access (OWA) in Microsoft Exchange Server 2013 PS1, 2013 Cumulative Update 10, 2013 Cumulative Update 11, and 2016 allows remote attackers to inject arbitrary web script or HTML via a crafted URL, aka "Exchange Spoofing Vulnerability."', 'Stack-based buffer overflow in Microsoft Excel 2000, 2002, and 2003, in Microsoft Office 2000 SP3 and other packages, allows user-assisted attackers to execute arbitrary code via an Excel file with a malformed record with a modified length value, which leads to memory corruption.', 'A certain Fedora patch for parse.c in sudo before 1.7.4p5-1.fc14 on Fedora 14 does not properly interpret a system group (aka %group) in the sudoers file during authorization decisions for a user who belongs to that group, which allows local users to leverage an applicable sudoers file and gain root privileges via a sudo command.  NOTE: this vulnerability exists because of a CVE-2009-0034 regression.', 'The forms-based authentication implementation in Active Directory Federation Services (ADFS) 3.0 in Microsoft Windows Server 2012 R2 allows remote attackers to cause a denial of service (daemon outage) via crafted data, aka "Microsoft Active Directory Federation Services Denial of Service Vulnerability."', 'ip_nat_pptp in the PPTP NAT helper (netfilter/ip_nat_helper_pptp.c) in Linux kernel 2.6.14, and other versions, allows local users to cause a denial of service (memory corruption or crash) via a crafted outbound packet that causes an incorrect offset to be calculated from pointer arithmetic when non-linear SKBs (socket buffers) are used.', 'Apache does not filter terminal escape sequences from its error logs, which could make it easier for attackers to insert those sequences into terminal emulators containing vulnerabilities related to escape sequences.', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2013. Notes: none.', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2016. Notes: none.', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2013. Notes: none.', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2013. Notes: none.', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2013. Notes: none.', 'Directory traversal vulnerability in Sun Kodak Color Management System (KCMS) library service daemon (kcms_server) allows remote attackers to read arbitrary files via the KCS_OPEN_PROFILE procedure.', 'Apple iTunes before 8.1 on Windows allows remote attackers to cause a denial of service (infinite loop) via a Digital Audio Access Protocol (DAAP) message with a crafted Content-Length header.', 'Multiple buffer overflows in libmcrypt before 2.5.5 allow attackers to cause a denial of service (crash).', 'Network Policy Server (NPS) in Microsoft Windows Server 2008 SP2 and R2 SP1 and Server 2012 Gold and R2 misparses username queries, which allows remote attackers to cause a denial of service (RADIUS authentication outage) via crafted requests, aka "Network Policy Server RADIUS Implementation Denial of Service Vulnerability."', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2013. Notes: none.', 'Buffer overflow in the RPC preprocessor for Snort 1.8 and 1.9.x before 1.9.1 allows remote attackers to execute arbitrary code via fragmented RPC packets.', 'ml85p, as included in the printer-drivers package for Mandrake Linux, allows local users to overwrite arbitrary files via a symlink attack on temporary files with predictable filenames of the form "mlg85p%d".', 'Unspecified vulnerability in context.py in Albatross web application toolkit before 1.33 allows remote attackers to execute arbitrary commands via unspecified vectors involving template files and the "handling of submitted form fields".', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2013. Notes: none.', 'The Windows Forms (aka WinForms) component in Microsoft .NET Framework 2.0 SP2, 3.5, 3.5.1, 4, and 4.5 does not properly restrict the privileges of a callback function during object creation, which allows remote attackers to execute arbitrary code via (1) a crafted XAML browser application (XBAP) or (2) a crafted .NET Framework application, aka "WinForms Callback Elevation Vulnerability."', 'Francesco Stablum tcpick 0.2.1 allows remote attackers to cause a denial of service (segmentation fault) via certain fragmented packets, possibly involving invalid headers and an attacker-controlled payload length.  NOTE: this issue might be a buffer overflow or overread.', 'Apache CouchDB 0.8.0 through 0.10.1 allows remote attackers to obtain sensitive information by measuring the completion time of operations that verify (1) hashes or (2) passwords.', 'Use-after-free vulnerability in Microsoft Office 2007 SP3, Excel 2007 SP3, PowerPoint 2007 SP3, Word 2007 SP3, Office 2010 SP2, Excel 2010 SP2, PowerPoint 2010 SP2, Word 2010 SP2, Office 2013 Gold and SP1, Word 2013 Gold and SP1, Office 2013 RT Gold and SP1, Word 2013 RT Gold and SP1, Excel Viewer, Office Compatibility Pack SP3, Word Automation Services on SharePoint Server 2010 SP2, Excel Services on SharePoint Server 2013 Gold and SP1, Word Automation Services on SharePoint Server 2013 Gold and SP1, Web Applications 2010 SP2, Office Web Apps Server 2010 SP2, Web Apps Server 2013 Gold and SP1, SharePoint Server 2007 SP3, Windows SharePoint Services 3.0 SP3, SharePoint Foundation 2010 SP2, SharePoint Server 2010 SP2, SharePoint Foundation 2013 Gold and SP1, and SharePoint Server 2013 Gold and SP1 allows remote attackers to execute arbitrary code via a crafted Office document, aka "Microsoft Office Component Use After Free Vulnerability." <a href="http://cwe.mitre.org/data/definitions/416.html">CWE-416: Use After Free</a>', 'Microsoft Internet Explorer 9 through 11 and Microsoft Edge misparse HTTP responses, which allows remote attackers to spoof web sites via a crafted URL, aka "Microsoft Browser Spoofing Vulnerability."', 'Imager (libimager-perl) before 0.50 allows user-assisted attackers to cause a denial of service (segmentation fault) by writing a 2- or 4-channel JPEG image (or a 2-channel TGA image) to a scalar, which triggers a NULL pointer dereference.', 'The ispell_op function in ee on FreeBSD 4.10 to 6.0 uses predictable filenames and does not confirm which file is being written, which allows local users to overwrite arbitrary files via a symlink attack when ee invokes ispell.', 'Microsoft Internet Explorer 5.01, 5.5, and 6 allows remote attackers to bypass the Kill bit settings for dangerous ActiveX controls via unknown vectors involving crafted HTML, which can expose the browser to attacks that would otherwise be prevented by the Kill bit setting. NOTE: CERT/CC claims that MS05-054 fixes this issue, but it is not described in MS05-054.', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2016. Notes: none.', 'Microsoft Excel 2007 SP3, PowerPoint 2007 SP3, Word 2007 SP3, Excel 2010 SP2, PowerPoint 2010 SP2, and Word 2010 SP2 allow remote attackers to execute arbitrary code via a crafted Office document, aka "Microsoft Word Local Zone Remote Code Execution Vulnerability."', 'Multiple cross-site scripting (XSS) vulnerabilities in the (1) examples and (2) ROOT web applications for Jakarta Tomcat 3.x through 3.3.1a allow remote attackers to insert arbitrary web script or HTML.', 'The kernel-mode driver in Microsoft Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8.1, Windows Server 2012 Gold and R2, Windows RT 8.1, and Windows 10 Gold and 1511 allows local users to gain privileges via a crafted application, aka "Win32k Elevation of Privilege Vulnerability," a different vulnerability than CVE-2016-0093, CVE-2016-0094, and CVE-2016-0095.', 'Microsoft Windows Vista SP2 and Server 2008 SP2 mishandle library loading, which allows local users to gain privileges via a crafted application, aka "Library Loading Input Validation Remote Code Execution Vulnerability."', 'Cross-site request forgery (CSRF) vulnerability in IBM Leads 7.x, 8.1.0 before 8.1.0.14, 8.2, 8.5.0 before 8.5.0.7.3, 8.6.0 before 8.6.0.8.1, 9.0.0 through 9.0.0.4, 9.1.0 before 9.1.0.6.1, and 9.1.1 before 9.1.1.0.2 allows remote authenticated users to hijack the authentication of customer accounts.', 'The PDF library in Microsoft Windows 8.1, Windows Server 2012 Gold and R2, Windows RT 8.1, and Windows 10 Gold and 1511 allows remote attackers to execute arbitrary code via a crafted PDF document, aka "Windows Remote Code Execution Vulnerability."', 'Microsoft Edge allows remote attackers to execute arbitrary code or cause a denial of service (memory corruption) via a crafted web site, aka "Microsoft Edge Memory Corruption Vulnerability," a different vulnerability than CVE-2016-0116, CVE-2016-0123, CVE-2016-0129, and CVE-2016-0130.', 'Microsoft Word 2007 SP3, Office 2010 SP2, Word 2010 SP2, Word 2013 SP1, Word 2013 RT SP1, Office Compatibility Pack SP3, Word Viewer, Word Automation Services on SharePoint Server 2010 SP2, Word Automation Services on SharePoint Server 2013 SP1, Office Web Apps 2010 SP2, and Office Web Apps Server 2013 SP1 allow remote attackers to execute arbitrary code via a crafted Office document, aka "Microsoft Office Memory Corruption Vulnerability."', 'SSH2 clients for VanDyke (1) SecureCRT 4.0.2 and 3.4.7, (2) SecureFX 2.1.2 and 2.0.4, and (3) Entunnel 1.0.2 and earlier, do not clear logon credentials from memory, including plaintext passwords, which could allow attackers with access to memory to steal the SSH credentials.', 'The USB Mass Storage Class driver in Microsoft Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8.1, Windows Server 2012 Gold and R2, Windows RT 8.1, and Windows 10 Gold and 1511 allows physically proximate attackers to execute arbitrary code by inserting a crafted USB device, aka "USB Mass Storage Elevation of Privilege Vulnerability."', 'parse_xml.cgi in Apple Darwin Streaming Administration Server 4.1.2 and QuickTime Streaming Server 4.1.1 allows remote attackers to list arbitrary directories.', 'Microsoft Excel 2010 SP2, Word for Mac 2011, and Excel Viewer allow remote attackers to execute arbitrary code via a crafted Office document, aka "Microsoft Office Memory Corruption Vulnerability."', 'Buffer overflow in secure locate (slocate) before 2.7 allows local users to execute arbitrary code via a long (1) -c or (2) -r command line argument.', 'Multiple cross-site scripting (XSS) vulnerabilities in Apache Jackrabbit before 1.5.2 allow remote attackers to inject arbitrary web script or HTML via the q parameter to (1) search.jsp or (2) swr.jsp.', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2016. Notes: none.', 'Microsoft .NET Framework 4.6 and 4.6.1 mishandles library loading, which allows local users to gain privileges via a crafted application, aka ".NET Framework Remote Code Execution Vulnerability."', 'McAfee ePolicy Orchestrator (ePO) 2.5.1 Patch 13 and 3.0 SP2a Patch 3 allows remote attackers to execute arbitrary commands via certain HTTP POST requests to the spipe/file handler on ePO TCP port 81.', 'Multiple format string vulnerabilities in HTTP Application Intelligence (AI) component in Check Point Firewall-1 NG-AI R55 and R54, and Check Point Firewall-1 HTTP Security Server included with NG FP1, FP2, and FP3 allows remote attackers to execute arbitrary code via HTTP requests that cause format string specifiers to be used in an error message, as demonstrated using the scheme of a URI.', 'Ember.js 1.0.x before 1.0.1, 1.1.x before 1.1.3, 1.2.x before 1.2.1, 1.3.x before 1.3.1, and 1.4.x before 1.4.0-beta.2 allows remote attackers to conduct cross-site scripting (XSS) attacks by leveraging an application using the "{{group}}" Helper and a crafted payload.', "The rxvt terminal emulator 2.7.8 and earlier allows attackers to modify the window title via a certain character escape sequence and then insert it back to the command line in the user's terminal, e.g. when the user views a file containing the malicious sequence, which could allow the attacker to execute arbitrary commands.", '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2013. Notes: none.', '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: The CNA or individual who requested this candidate did not associate it with any vulnerability during 2013. Notes: none.', 'Red Hat JBoss Operations Network (JON) before 3.0.1 uses 0777 permissions for the root directory when installing a remote client, which allows local users to read or modify subdirectories and files within the root directory, as demonstrated by obtaining JON credentials.', 'The kernel-mode drivers in Microsoft Windows Vista SP2, Windows Server 2008 SP2 and R2 SP1, Windows 7 SP1, Windows 8.1, Windows Server 2012 Gold and R2, Windows RT 8.1, and Windows 10 Gold and 1511 allow local users to gain privileges via a crafted application, aka "Win32k Elevation of Privilege Vulnerability," a different vulnerability than CVE-2016-0171, CVE-2016-0173, and CVE-2016-0196.']
    vendor = [['microsoft'], ['microsoft'], ['microsoft'], ['microsoft'], ['gnu'], ['microsoft'], ['microsoft'], ['miloslav_trmac'], ['microsoft'], ['microsoft'], ['microsoft'], ['mozilla'], ['microsoft'], ['bmv'], ['microsoft'], [], ['microsoft'], [], ['microsoft'], ['microsoft'], ['apache'], ['microsoft'], ['apache'], ['microsoft'], ['microsoft'], ['microsoft'], ['microsoft'], [], ['microsoft'], ['microsoft'], ['microsoft'], ['apache'], ['linux'], ['joomla'], ['microsoft'], ['viewvc'], ['microsoft'], ['microsoft'], ['kde'], ['microsoft'], ['microsoft'], ['microsoft'], ['microsoft'], ['microsoft'], ['linux'], ['microsoft'], ['microsoft'], ['microsoft'], ['microsoft'], ['todd_miller'], ['microsoft'], ['linux'], ['apache'], [], [], [], [], [], ['sun'], ['apple'], ['mcrypt'], ['microsoft'], [], ['snort'], ['rildo_pragana'], ['albatross'], [], ['microsoft'], ['francesco_stablum'], ['apache'], ['microsoft'], ['microsoft'], ['tony_cook'], ['freebsd'], ['microsoft'], [], ['microsoft'], ['apache'], ['microsoft'], ['microsoft'], ['ibm'], ['microsoft'], ['microsoft'], ['microsoft'], ['van_dyke_technologies'], ['microsoft'], ['apple'], ['microsoft'], ['slocate'], ['apache'], [], ['microsoft'], ['mcafee'], ['checkpoint'], ['emberjs'], ['rxvt'], [], [], ['redhat'], ['microsoft']]
    product = [['windows_xp'], ['office'], ['windows_7', 'windows_server_2003', 'windows_server_2008', 'windows_vista', 'windows_xp'], ['windows_me', 'windows_xp'], ['bash'], ['windows_2000', 'windows_2000_terminal_services', 'windows_98', 'windows_98se', 'windows_me', 'windows_nt', 'windows_xp'], ['.net_framework'], ['libuser'], ['windows_8', 'windows_8.1', 'windows_rt', 'windows_rt_8.1', 'windows_server_2012'], ['windows_7', 'windows_8', 'windows_8.1', 'windows_rt', 'windows_rt_8.1', 'windows_server_2008', 'windows_server_2012'], ['windows_7', 'windows_server_2003', 'windows_server_2008', 'windows_vista', 'windows_xp'], ['bugzilla'], ['windows_7', 'windows_server_2003', 'windows_server_2008', 'windows_vista', 'windows_xp'], ['bmv'], ['windows_server_2003', 'windows_server_2008', 'windows_vista', 'windows_xp'], [], ['office', 'windows_2003_server', 'windows_xp'], [], ['windows_7', 'windows_8', 'windows_8.1', 'windows_rt', 'windows_rt_8.1', 'windows_server_2003', 'windows_server_2008', 'windows_server_2012', 'windows_vista'], ['windows_7', 'windows_8', 'windows_8.1', 'windows_rt', 'windows_rt_8.1', 'windows_server_2003', 'windows_server_2008', 'windows_server_2012', 'windows_vista'], ['camel'], ['windows_7', 'windows_8', 'windows_8.1', 'windows_rt', 'windows_rt_8.1', 'windows_server_2003', 'windows_server_2008', 'windows_server_2012'], ['tomcat'], ['anti-cross_site_scripting_library'], ['.net_framework'], ['windows_7', 'windows_8', 'windows_8.1', 'windows_server_2003', 'windows_server_2008', 'windows_server_2012', 'windows_vista'], ['windows_server_2003', 'windows_server_2008', 'windows_server_2012'], [], ['windows_server_2003', 'windows_xp'], ['edge'], ['windows_2003_server', 'windows_xp'], ['camel'], ['linux_kernel'], ['com_search'], ['windows_10', 'windows_7', 'windows_8', 'windows_8.1', 'windows_rt', 'windows_rt_8.1', 'windows_server_2008', 'windows_server_2012', 'windows_vista'], ['viewvc'], ['windows_10', 'windows_7', 'windows_8', 'windows_8.1', 'windows_rt', 'windows_rt_8.1', 'windows_server_2008', 'windows_server_2012', 'windows_vista'], ['windows_7', 'windows_8', 'windows_8.1', 'windows_rt', 'windows_rt_8.1', 'windows_server_2008', 'windows_server_2012', 'windows_vista'], ['kde'], ['ie'], ['windows_7', 'windows_server_2003', 'windows_server_2008', 'windows_vista', 'windows_xp'], ['sharepoint_server', 'sharepoint_foundation'], ['system_center_operations_manager'], ['.net_framework'], ['linux_kernel'], ['exchange_server'], ['exchange_server'], ['exchange_server'], ['office'], ['sudo'], ['windows_server_2012'], ['linux_kernel'], ['http_server'], [], [], [], [], [], ['solaris', 'sunos'], ['itunes'], ['libmcrypt'], ['windows_server_2008', 'windows_server_2012'], [], ['snort'], ['ml85p'], ['albatross'], [], ['.net_framework'], ['tcpick'], ['couchdb'], ['excel', 'excel_viewer', 'office', 'office_compatibility_pack', 'office_web_apps_server', 'powerpoint', 'sharepoint_foundation', 'sharepoint_server', 'sharepoint_services', 'web_applications', 'word'], ['edge', 'internet_explorer'], ['imager'], ['freebsd'], ['ie'], [], ['excel', 'powerpoint', 'word'], ['tomcat'], ['windows_10', 'windows_7', 'windows_8.1', 'windows_rt_8.1', 'windows_server_2008', 'windows_server_2012', 'windows_vista'], ['windows_server_2008', 'windows_vista'], ['leads'], ['windows_10', 'windows_8.1', 'windows_rt_8.1', 'windows_server_2012'], ['edge'], ['office', 'office_compatibility_pack', 'office_web_apps_server', 'sharepoint_server', 'word', 'word_viewer'], ['entunnel', 'securecrt', 'securefx'], ['windows_10', 'windows_7', 'windows_8.1', 'windows_rt_8.1', 'windows_server_2008', 'windows_vista'], ['darwin_streaming_server', 'quicktime_streaming_server'], ['excel', 'excel_viewer', 'word_for_mac'], ['slocate'], ['jackrabbit'], [], ['.net_framework'], ['epolicy_orchestrator'], ['firewall-1'], ['ember.js'], ['rxvt'], [], [], ['jboss_operations_network'], ['windows_10', 'windows_7', 'windows_8.1', 'windows_rt_8.1', 'windows_server_2008', 'windows_server_2012', 'windows_vista']]
    summary_list = []
    vendor_list = []
    product_list = []
    postags_list = []
    ne_chunked_tags_list = []
    merge_list = []

    for i in range(len(summary)):

        summary_token = summary_deal(summary[i],1)

        vendor_lable = ' '.join(vendor[i])  #获取单个vendor小列表
        list_summary_x, list_vendor_y = summary_lable_process(summary_token, vendor_lable,0)
        for i1 in list_summary_x:
            summary_list.append(i1)
        summary_list.append('.') #在每条summary结尾做标记
        for i2 in list_vendor_y:
            vendor_list.append(i2)

        tmp_product_lable = product[i]

        #判断product_lable是否有多个，若有多个就进行特殊处理
        if len(tmp_product_lable)>1:
            c = 0
            for tmp in tmp_product_lable:
                tmp = tmp.replace('_',' ')
                product_lable = tmp.replace('-',' ')
                # print(product_lable)
                list_summary_x, list_vendor_y = summary_lable_process(summary_token, product_lable,1)
                list_product_t = list_vendor_y
                # print(list_product_t)
                #将多个product_lable得到的标签合并为一个列表
                if c==0:
                    l1 = list_product_t
                    c=c+1
                    continue
                else:
                    l2 = list_product_t
                    a = mergelist(l1,l2)
                    l1 = a
                    list_product_y = l1
                # print(list_product_y)
                #最终结果仍为list_product_y
        else:
            tmp_product_lable = ' '.join(product[i])
            tmp_product_lable = tmp_product_lable.replace('_',' ') #将product中的下划线去除
            product_lable = tmp_product_lable.replace('-',' ')
            # print(product_lable)
            list_summary_x, list_product_y = summary_lable_process(summary_token, product_lable,1)
            # print(list_product_y)

        list_merge = mergelist(list_vendor_y,list_product_y)
        for i9 in list_merge:
            merge_list.append(i9)
        merge_list.append('O')
        # print(merge_list)

        #获取词性和chunking标签
        list_postags = postags(summary_token,1)
        # print(list_postags)
        for i5 in list_postags:
            postags_list.append(i5)
        postags_list.append('.')

        list_chunkingtags = postags(summary_token,2)
        for i7 in list_chunkingtags:
            ne_chunked_tags_list.append(i7)
        ne_chunked_tags_list.append('O')

    #检查长度
    # print(len(summary_list))
    # print(len(merge_list))
    # print(len(postags_list))
    # print(postags_list)
    # print(len(chunkingtags_list))

    #使用np，将之前处理好的多个列表合并为矩阵并转置(注意，每个列表长度应该保持一致)
    tmp_array = np.vstack((summary_list,merge_list)) #summary_list,postags_list,merge_list,ne_chunked_tags_list
    lable_array = tmp_array.T
    np.savetxt("c1.txt",lable_array,fmt="%s") #将转制后的矩阵存为txt
    print(lable_array)

    c = 0# 结尾标志位
    # cx.txt为临时文件，不含结尾分割无需保存
    with open("testset20000-1.txt", "w", encoding="utf-8") as f1:
        with open("c1.txt", "r+", encoding="utf-8") as f:
            data = f.readlines()
        for line in data:
            # print(line)
            if line == '. O\n':
                f1.write(line)
                c = 1
            elif c == 1:
                f1.write('\r')
                c = 0
            else:
                f1.write(line)

        # # 含符号conll2003数据集规范的结尾标志
        # for line in data:
        #     # print(line)
        #     if line == '. O\n':
        #         f1.write(line)
        #         c = 1
        #     elif c == 1:
        #         f1.write("\n"+"\n"+"-DOCSTART- -X- O O"+"\n"+"\n")
        #         c = 0
        #     else:
        #         f1.write(line)
