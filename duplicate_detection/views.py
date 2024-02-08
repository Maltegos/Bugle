from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from .forms import BugReportForm
from .models import BugReport
# uncomment trainer to call code implemented with spaCy
# from .trainer import get_similar_bugs
# from .duplicate_finder import get_similar_bugs
import time
from .sbert_duplicate_finder import get_similar_bugs


# Create your views here.

# submit new issue description
def home(request):
    if request.POST:
        form = BugReportForm(request.POST)
        if form.is_valid():
            form.save()
    return render(request, 'upload.html', {'form': BugReportForm})


# def upload(request):
#   bugs = BugReport.objects.all()
#  return render(request, "upload.html", {"bugs": bugs})

# def upload(request):
#    print("In upload")
#    print(request)
#    if request.POST:
#        bugs = BugReportForm(request.POST)
#        if bugs.is_valid():
#            print("It is savsed!!!")
#            bugs.save()
#
#    return render(request, "upload.html", {"bugs": BugReportForm})

def upload(request):
    if request.POST:
        form = BugReportForm(request.POST)
        if form.is_valid():
            input_bug = request.POST.get("name")
            form.save()
            similar1 = get_similar(input_bug)
            similar = [sub.replace('nan', '') for sub in similar1]
            return render(request, 'ajax.html', {"similar": similar, "form": BugReportForm, "input_bug": input_bug})
    else:
        similar = []
        input_bug = ""
        return render(request, 'ajax.html', {"similar": similar, "form": BugReportForm, "input_bug": input_bug})


def create(request):
    backup = []
    # if request.method == 'POST':
    #    description = request.POST['description']
    #    new_description = BugReport(description=description)
    #    new_description.save()
    #    success = 'User '+description+' created successfully'
    #    print("Success:::: " + success)
    #    return HttpResponse(success)

    if request.POST:
        form = BugReportForm(request.POST)
        if form.is_valid():
            result = []
            input_bug = request.POST.get("description")
            form.save()
            similar = get_similar(input_bug)
            similar_copy = []
            for index in range(len(similar)):
                dict = similar[index][1].get("description")
                if str(dict) == "nan":
                    similar_copy.append(similar[index][1].get("issue_id"))
            for i in range(len(similar)):
                if similar[i][1].get("issue_id") in similar_copy:
                    continue
                else:
                    result.append(similar[i])
            return JsonResponse({'similar': result})
    else:
        return JsonResponse({'backup': backup})


def get_similar(similar):
    result = get_similar_bugs(similar)
    return result
